#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.daedl_project.artifacts import (
    create_run_directory,
    environment_record,
    save_predictions,
    write_json,
)
from src.daedl_project.data import CIFAR10_MEAN, CIFAR10_STD, build_cifar10_loaders
from src.daedl_project.evaluation import evaluate_model, fit_daedl_density
from src.daedl_project.model import CifarResNet18
from src.daedl_project.plots import plot_reliability_diagram
from src.daedl_project.reproducibility import DEFAULT_SEED, choose_device, seed_everything
from src.daedl_project.training import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a clearly labelled Phase-2 pilot")
    parser.add_argument("--model", choices=["softmax", "daedl"], required=True)
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"], default="mps")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-size", type=int, default=5_000)
    parser.add_argument("--validation-size", type=int, default=1_000)
    parser.add_argument("--test-size", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--regularization", type=float, default=5e-2)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--base-width", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    return parser.parse_args()


def state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.epochs not in (1, 2):
        raise ValueError("Phase-2 pilot is intentionally limited to 1-2 epochs")
    seed_everything(args.seed)
    choice = choose_device(args.device, allow_cpu_fallback=args.allow_cpu_fallback)
    device = torch.device(choice.effective)
    print(f"Device requested={choice.requested}, effective={choice.effective}; no silent fallback")

    config = vars(args).copy()
    config.update(
        {
            "phase": "pilot_only",
            "normalization_mean": CIFAR10_MEAN,
            "normalization_std": CIFAR10_STD,
            "training_augmentation": ["random_crop_32_padding_4", "random_horizontal_flip"],
            "validation_test_augmentation": [],
            "architecture": "compact_cifar_resnet18",
            "spectral_normalization": args.model == "daedl",
            "density_device": "cpu",
            "density_dtype": "float64",
        }
    )
    run_dir = create_run_directory(args.output_root, args.model, args.seed)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "environment.json", environment_record(choice.__dict__))

    loaders = build_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        split_seed=args.split_seed,
        loader_seed=args.seed,
        pilot_train_size=args.train_size,
        pilot_validation_size=args.validation_size,
        pilot_test_size=args.test_size,
        num_workers=args.num_workers,
    )
    np.savez_compressed(
        run_dir / "split_indices.npz",
        train=loaders.train_indices,
        validation=loaders.validation_indices,
        test=loaders.test_indices,
    )
    first_images, first_labels, _ = next(iter(loaders.train))
    if first_images.shape[1:] != (3, 32, 32):
        raise AssertionError(f"unexpected CIFAR tensor shape: {tuple(first_images.shape)}")
    if set(first_labels.tolist()) - set(range(10)):
        raise AssertionError("invalid class label")

    model_kwargs = {
        "num_classes": 10,
        "embedding_dim": args.embedding_dim,
        "base_width": args.base_width,
        "spectral_normalization": args.model == "daedl",
    }
    model = CifarResNet18(**model_kwargs)
    training = train_model(
        model=model,
        model_type=args.model,
        train_loader=loaders.train,
        validation_loader=loaders.validation,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        regularization=args.regularization,
    )
    pd.DataFrame(training.history).to_csv(run_dir / "training_history.csv", index=False)

    checkpoint = {
        "phase": "pilot_only",
        "model_type": args.model,
        "model_kwargs": model_kwargs,
        "model_state": training.best_state,
        "seed": args.seed,
        "split_seed": args.split_seed,
    }
    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    # Explicit load test: reconstructed weights must have the same digest.
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    reloaded_model = CifarResNet18(**loaded["model_kwargs"])
    reloaded_model.load_state_dict(loaded["model_state"])
    checkpoint_verified = state_digest(reloaded_model.state_dict()) == state_digest(training.best_state)
    if not checkpoint_verified:
        raise AssertionError("checkpoint round-trip changed model weights")
    model.load_state_dict(training.best_state)
    model.to(device)

    gda_state = None
    density_summary = None
    if args.model == "daedl":
        gda_state = fit_daedl_density(model, loaders.density, device)
        density_summary = {
            "device": gda_state.means.device.type,
            "dtype": str(gda_state.means.dtype),
            "embedding_dim": int(gda_state.means.shape[1]),
            "log_density_min": gda_state.log_density_min,
            "log_density_max": gda_state.log_density_max,
            "jitter_by_class": gda_state.jitter_by_class,
        }
        torch.save(
            {
                "means": gda_state.means,
                "cholesky": gda_state.cholesky,
                "log_weights": gda_state.log_weights,
                "log_density_min": gda_state.log_density_min,
                "log_density_max": gda_state.log_density_max,
                "jitter_by_class": gda_state.jitter_by_class,
            },
            run_dir / "gda_state.pt",
        )
        write_json(run_dir / "density_summary.json", density_summary)

    metrics, rows = evaluate_model(model, loaders.test, device, args.model, gda_state)
    metrics.update(
        {
            "phase": "PILOT_ONLY_NOT_FINAL",
            "model": args.model,
            "train_size": len(loaders.train.dataset),
            "validation_size": len(loaders.validation.dataset),
            "test_size": len(loaders.test.dataset),
            "epochs": args.epochs,
            "runtime_seconds": training.runtime_seconds,
            "checkpoint_round_trip_verified": checkpoint_verified,
            "final_epoch": training.history[-1],
            "density": density_summary,
        }
    )
    write_json(run_dir / "metrics.json", metrics)
    save_predictions(run_dir / "predictions.csv", rows)

    probabilities = np.asarray(
        [[row[f"probability_{class_index}"] for class_index in range(10)] for row in rows]
    )
    labels = np.asarray([row["ground_truth"] for row in rows])
    plot_reliability_diagram(
        probabilities,
        labels,
        run_dir / "pilot_reliability.png",
        title=f"PILOT ONLY: {args.model}",
    )
    freeze = subprocess.run(
        [str(Path(".venv/bin/python")), "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    ).stdout
    (run_dir / "pip_freeze.txt").write_text(freeze)
    (run_dir / "logs.txt").write_text(
        "PILOT ONLY - not for final report results.\n"
        f"device_requested={choice.requested}\n"
        f"device_effective={choice.effective}\n"
        f"checkpoint_round_trip_verified={checkpoint_verified}\n"
    )
    print(json.dumps({"run_dir": str(run_dir), "metrics": metrics}, indent=2, default=str))


if __name__ == "__main__":
    main()

