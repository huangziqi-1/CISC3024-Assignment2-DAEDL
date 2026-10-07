#!/usr/bin/env python3
"""Run one frozen final model from training through clean and SVHN evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision import datasets

from daedl_project.artifacts import create_run_directory, environment_record, save_predictions, write_json
from daedl_project.data import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    build_cifar10_loaders,
    build_svhn_loader,
)
from daedl_project.evaluation import (
    embedding_density_diagnostics,
    evaluate_model,
    extract_embeddings,
    ood_metrics_from_rows,
    score_ood_loader,
)
from daedl_project.gda import density_score, fit_gda, log_density
from daedl_project.metrics import reliability_data
from daedl_project.model import CifarResNet18
from daedl_project.reproducibility import DEFAULT_SEED, choose_device, seed_everything
from daedl_project.training import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one frozen final core/replication experiment")
    parser.add_argument("--model", choices=["softmax", "daedl"], required=True)
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"], default="mps")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--regularization", type=float, default=5e-2)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--base-width", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--phase", choices=["FINAL_CORE", "FINAL_REPLICATION"], default="FINAL_CORE")
    parser.add_argument("--skip-individual-figures", action="store_true")
    return parser.parse_args()


def state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def score_summary(rows: list[dict], key: str) -> dict[str, float]:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    return {"mean": float(values.mean()), "median": float(np.median(values))}


def save_figure_inputs(run_dir: Path, test_rows: list[dict], ood_rows: list[dict]) -> None:
    pd.DataFrame(
        {
            "dataset": ["CIFAR-10"] * len(test_rows) + ["SVHN"] * len(ood_rows),
            "normalized_predictive_entropy": [row["normalized_predictive_entropy"] for row in test_rows]
            + [row["normalized_predictive_entropy"] for row in ood_rows],
        }
    ).to_csv(run_dir / "figure_input_ood_entropy.csv", index=False)
    pd.DataFrame(
        {
            "correctness": ["correct" if row["correct"] else "incorrect" for row in test_rows],
            "normalized_predictive_entropy": [row["normalized_predictive_entropy"] for row in test_rows],
        }
    ).to_csv(run_dir / "figure_input_error_entropy.csv", index=False)
    probabilities = np.asarray(
        [[row[f"probability_{class_index}"] for class_index in range(10)] for row in test_rows]
    )
    labels = np.asarray([row["ground_truth"] for row in test_rows])
    bins = reliability_data(probabilities, labels, n_bins=15)
    pd.DataFrame(
        {
            "lower": bins.lower,
            "upper": bins.upper,
            "count": bins.count,
            "mean_confidence": bins.mean_confidence,
            "accuracy": bins.accuracy,
        }
    ).to_csv(run_dir / "figure_input_reliability.csv", index=False)


def plot_analysis_figures(run_dir: Path, model_name: str) -> None:
    history = pd.read_csv(run_dir / "training_history.csv")
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(history.epoch, history.train_loss)
    axes[0].set(title="Training objective", xlabel="Epoch", ylabel="Mean objective")
    axes[1].plot(history.epoch, history.train_accuracy, label="Train")
    axes[1].plot(history.epoch, history.validation_accuracy, label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()
    axes[2].plot(history.epoch, history.validation_nll)
    axes[2].set(title="Validation NLL", xlabel="Epoch", ylabel="NLL")
    figure.suptitle(f"FINAL_CORE {model_name}")
    figure.tight_layout()
    figure.savefig(run_dir / "training_validation_curves.png", dpi=160)
    plt.close(figure)

    reliability = pd.read_csv(run_dir / "figure_input_reliability.csv")
    populated = reliability[reliability["count"] > 0]
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot([0, 1], [0, 1], "--", color="black", linewidth=1, label="Ideal")
    axis.plot(populated.mean_confidence, populated.accuracy, "o-", label=model_name)
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Accuracy", title="Reliability")
    axis.legend()
    figure.tight_layout()
    figure.savefig(run_dir / "reliability_diagram.png", dpi=160)
    plt.close(figure)

    for source, column, output, title in (
        ("figure_input_error_entropy.csv", "correctness", "error_entropy_distribution.png", "Correct vs incorrect"),
        ("figure_input_ood_entropy.csv", "dataset", "ood_entropy_distribution.png", "CIFAR-10 vs SVHN"),
    ):
        data = pd.read_csv(run_dir / source)
        figure, axis = plt.subplots(figsize=(6, 4))
        for group, values in data.groupby(column, sort=False):
            axis.hist(values.normalized_predictive_entropy, bins=40, density=True, alpha=0.45, label=group)
        axis.set(xlabel="Normalized predictive entropy", ylabel="Density", title=title)
        axis.legend()
        figure.tight_layout()
        figure.savefig(run_dir / output, dpi=160)
        plt.close(figure)


def choose_representatives(test_rows: list[dict], ood_rows: list[dict]) -> list[dict]:
    correct = [row for row in test_rows if row["correct"]]
    incorrect = [row for row in test_rows if not row["correct"]]
    rules = [
        ("highest-confidence correct", max(correct, key=lambda r: (r["max_confidence"], -r["sample_index"]))),
        ("highest-entropy correct", max(correct, key=lambda r: (r["normalized_predictive_entropy"], -r["sample_index"]))),
        ("lowest-index incorrect", min(incorrect, key=lambda r: r["sample_index"])),
        ("highest-entropy incorrect", max(incorrect, key=lambda r: (r["normalized_predictive_entropy"], -r["sample_index"]))),
    ]
    ood_entropy = np.asarray([row["normalized_predictive_entropy"] for row in ood_rows])
    median = float(np.median(ood_entropy))
    representative_ood = min(
        ood_rows, key=lambda r: (abs(r["normalized_predictive_entropy"] - median), r["sample_index"])
    )
    result = []
    for rule, row in rules:
        result.append({"dataset": "CIFAR-10", "rule": rule, **row})
    result.append({"dataset": "SVHN", "rule": "closest-to-median OOD entropy", **representative_ood})
    return result


def plot_representatives(run_dir: Path, data_dir: Path, selected: list[dict]) -> None:
    cifar = datasets.CIFAR10(data_dir, train=False, download=False)
    svhn = datasets.SVHN(data_dir, split="test", download=False)
    figure, axes = plt.subplots(1, len(selected), figsize=(15, 3.4))
    for axis, item in zip(axes, selected):
        image, _ = (cifar if item["dataset"] == "CIFAR-10" else svhn)[int(item["sample_index"])]
        axis.imshow(image)
        axis.axis("off")
        axis.set_title(
            f"{item['dataset']}\n{item['rule']}\npred={item['predicted_label']}, H={item['normalized_predictive_entropy']:.3f}",
            fontsize=8,
        )
    figure.tight_layout()
    figure.savefig(run_dir / "representative_prediction_grid.png", dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.epochs != 30 or args.seed != args.split_seed:
        raise ValueError("final runs require 30 epochs and matching training/split seeds")
    if args.phase == "FINAL_CORE" and args.seed != 3024:
        raise ValueError("FINAL_CORE is locked to seed 3024")
    if args.phase == "FINAL_REPLICATION" and args.seed not in (3025, 3026):
        raise ValueError("FINAL_REPLICATION is locked to seed 3025 or 3026")
    seed_everything(args.seed)
    choice = choose_device(args.device, allow_cpu_fallback=False)
    training_device = torch.device(choice.effective)
    run_dir = args.run_dir or create_run_directory(
        args.output_root, args.model, args.seed,
        prefix="final_core" if args.phase == "FINAL_CORE" else "final_replication"
    )
    run_dir.mkdir(parents=True, exist_ok=False) if args.run_dir else None
    (run_dir / args.phase).write_text(f"{args.phase} seed-{args.seed} experiment\n")

    model_kwargs = {
        "num_classes": 10,
        "embedding_dim": args.embedding_dim,
        "base_width": args.base_width,
        "spectral_normalization": True,
    }
    config = vars(args).copy()
    config.update(
        {
            "phase": args.phase,
            "train_size": 45_000,
            "validation_size": 5_000,
            "test_size": 10_000,
            "svhn_evaluation_size": 10_000,
            "normalization_mean": CIFAR10_MEAN,
            "normalization_std": CIFAR10_STD,
            "training_augmentation": ["random_crop_32_padding_4", "random_horizontal_flip"],
            "architecture": "compact_cifar_resnet18",
            "model_kwargs": model_kwargs,
            "spectral_normalization": {
                "enabled": True,
                "identical_for_softmax_and_daedl": True,
                "layers": "all convolutional layers, shortcut projections, embedding linear layer, classifier linear layer",
            },
            "checkpoint_selection": "minimum validation NLL; test set never consulted",
            "training_probability_for_validation_nll": "softmax(logits), exactly exp(logits)/sum(exp(logits)) for DAEDL before density scaling",
            "daedl_loss_reduction": "mean of complete per-example objective; data and KL terms scaled together",
            "gradient_clip_max_norm": 10.0,
            "density_device": "cpu",
            "density_dtype": "float64",
            "post_training_evaluation_device": "cpu",
            "post_training_cpu_reason": "PyTorch 2.14.1 MPS produced non-finite outputs for a reloaded spectral-normalized pilot checkpoint; both final models use the same explicit CPU evaluation policy",
        }
    )
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "environment.json", environment_record(choice.__dict__))

    loaders = build_cifar10_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        split_seed=args.split_seed,
        loader_seed=args.seed,
        num_workers=args.num_workers,
    )
    np.savez_compressed(
        run_dir / "split_indices.npz",
        train=loaders.train_indices,
        validation=loaders.validation_indices,
        test=loaders.test_indices,
    )
    model = CifarResNet18(**model_kwargs)
    pipeline_started = time.perf_counter()
    training = train_model(
        model,
        args.model,
        loaders.train,
        loaders.validation,
        training_device,
        args.epochs,
        args.learning_rate,
        args.weight_decay,
        args.regularization,
        minimum_learning_rate=args.minimum_learning_rate,
        progress_log_path=run_dir / "training_progress.jsonl",
    )
    pd.DataFrame(training.history).to_csv(run_dir / "training_history.csv", index=False)

    checkpoint_base = {
        "phase": args.phase,
        "model_type": args.model,
        "model_kwargs": model_kwargs,
        "seed": args.seed,
        "split_seed": args.split_seed,
    }
    torch.save(
        {**checkpoint_base, "checkpoint_role": "best_validation_nll", "selected_epoch": training.best_epoch,
         "validation_nll": training.best_validation_nll, "model_state": training.best_state},
        run_dir / "best_validation_nll_checkpoint.pt",
    )
    torch.save(
        {**checkpoint_base, "checkpoint_role": "final_epoch", "epoch": args.epochs,
         "model_state": training.final_state},
        run_dir / "final_checkpoint.pt",
    )
    loaded = torch.load(run_dir / "best_validation_nll_checkpoint.pt", map_location="cpu", weights_only=True)
    checkpoint_verified = state_digest(loaded["model_state"]) == state_digest(training.best_state)
    if not checkpoint_verified:
        raise AssertionError("best checkpoint round trip changed weights")
    model.load_state_dict(loaded["model_state"])

    # Both methods use the identical spectral-normalized architecture. A
    # reloaded spectral-normalized pilot checkpoint produced non-finite outputs
    # on PyTorch 2.14.1 MPS, so evaluate both selected checkpoints on CPU.
    evaluation_device = torch.device("cpu")
    model.to(evaluation_device)
    density_runtime = 0.0
    gda_state = None
    density_diagnostics = None
    if args.model == "daedl":
        density_started = time.perf_counter()
        embeddings, density_labels, density_indices = extract_embeddings(model, loaders.density, evaluation_device)
        if not np.array_equal(np.sort(density_indices.numpy()), np.sort(loaders.train_indices)):
            raise AssertionError("density embeddings are not the clean full training split")
        density_diagnostics = embedding_density_diagnostics(embeddings, density_labels)
        if density_diagnostics["nonfinite_fraction"] != 0.0:
            raise FloatingPointError("non-finite full-training embedding")
        gda_state = fit_gda(embeddings, density_labels, num_classes=10)
        density_diagnostics.update(
            {
                "feature_extraction_device": "cpu",
                "gda_device": "cpu",
                "gda_dtype": "torch.float64",
                "jitter_by_class": gda_state.jitter_by_class,
                "training_log_density_min": gda_state.log_density_min,
                "training_log_density_max": gda_state.log_density_max,
            }
        )
        training_log_density = log_density(embeddings, gda_state)
        training_density_score = density_score(training_log_density, gda_state)
        pd.DataFrame(
            {
                "sample_index": density_indices.numpy(),
                "label": density_labels.numpy(),
                "log_feature_density": training_log_density.numpy(),
                "density_score": training_density_score.numpy(),
            }
        ).sort_values("sample_index").to_csv(run_dir / "training_density_values.csv", index=False)
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
        write_json(run_dir / "density_diagnostics.json", density_diagnostics)
        density_runtime = time.perf_counter() - density_started

    clean_started = time.perf_counter()
    test_metrics, test_rows = evaluate_model(model, loaders.test, evaluation_device, args.model, gda_state)
    clean_runtime = time.perf_counter() - clean_started
    test_metrics.update(
        {
            "phase": args.phase,
            "model": args.model,
            "selected_epoch": training.best_epoch,
            "selected_checkpoint_validation_nll": training.best_validation_nll,
            "checkpoint_round_trip_verified": checkpoint_verified,
            "test_count": len(test_rows),
        }
    )
    save_predictions(run_dir / "test_predictions.csv", test_rows)
    write_json(run_dir / "test_metrics.json", test_metrics)

    svhn_loader = build_svhn_loader(
        args.data_dir, args.batch_size, args.seed, subset_size=10_000, num_workers=args.num_workers
    )
    ood_started = time.perf_counter()
    ood_rows = score_ood_loader(model, svhn_loader, evaluation_device, args.model, gda_state)
    ood_metrics = ood_metrics_from_rows(test_rows, ood_rows, "normalized_predictive_entropy")
    ood_metrics.update(
        {
            "phase": args.phase,
            "positive_class": "OOD",
            "primary_score": "normalized_predictive_entropy",
            "higher_score_meaning": "more OOD-like",
            "id_count": len(test_rows),
            "ood_count": len(ood_rows),
            "id_uncertainty": score_summary(test_rows, "normalized_predictive_entropy"),
            "ood_uncertainty": score_summary(ood_rows, "normalized_predictive_entropy"),
        }
    )
    if args.model == "daedl":
        for row in test_rows + ood_rows:
            row["density_uncertainty"] = 1.0 - row["density_score"]
        secondary = ood_metrics_from_rows(test_rows, ood_rows, "density_uncertainty")
        secondary.update(
            {
                "score": "1-density_score",
                "higher_score_meaning": "more OOD-like",
                "id_uncertainty": score_summary(test_rows, "density_uncertainty"),
                "ood_uncertainty": score_summary(ood_rows, "density_uncertainty"),
            }
        )
        ood_metrics["secondary_density_uncertainty"] = secondary
    save_predictions(run_dir / "test_predictions.csv", test_rows)
    save_predictions(run_dir / "svhn_predictions.csv", ood_rows)
    write_json(run_dir / "ood_metrics.json", ood_metrics)
    ood_runtime = time.perf_counter() - ood_started

    if not args.skip_individual_figures:
        save_figure_inputs(run_dir, test_rows, ood_rows)
        plot_analysis_figures(run_dir, args.model)
        selected = choose_representatives(test_rows, ood_rows)
        write_json(run_dir / "representative_selection.json", selected)
        plot_representatives(run_dir, args.data_dir, selected)

    runtime = {
        "training_seconds": training.runtime_seconds,
        "density_fitting_and_diagnostics_seconds": density_runtime,
        "clean_evaluation_seconds": clean_runtime,
        "ood_evaluation_seconds": ood_runtime,
        "total_pipeline_seconds": time.perf_counter() - pipeline_started,
        "training_device": training_device.type,
        "evaluation_device": evaluation_device.type,
    }
    write_json(run_dir / "runtime.json", runtime)
    write_json(
        run_dir / "run_summary.json",
        {
            "phase": args.phase,
            "model": args.model,
            "epochs_completed": len(training.history),
            "selected_epoch": training.best_epoch,
            "selected_validation_nll": training.best_validation_nll,
            "final_train_accuracy": training.history[-1]["train_accuracy"],
            "best_validation_accuracy": max(row["validation_accuracy"] for row in training.history),
            "test_metrics": test_metrics,
            "ood_metrics": ood_metrics,
            "density_diagnostics": density_diagnostics,
            "runtime": runtime,
        },
    )
    freeze = subprocess.run(
        [str(Path(".venv/bin/python")), "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    ).stdout
    (run_dir / "pip_freeze.txt").write_text(freeze)
    (run_dir / "COMPLETED").write_text(f"{args.phase} pipeline completed successfully\n")
    print(json.dumps({"run_dir": str(run_dir), "summary": str(run_dir / "run_summary.json")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
