#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from daedl_project.data import build_cifar10_loaders
from daedl_project.evaluation import extract_embeddings
from daedl_project.model import CifarResNet18
from daedl_project.reproducibility import choose_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--device", choices=["mps", "cpu"], required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.run_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    model = CifarResNet18(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    choice = choose_device(args.device)
    device = torch.device(choice.effective)
    model.to(device)
    loaders = build_cifar10_loaders(
        "data", pilot_train_size=5_000, pilot_validation_size=1_000, pilot_test_size=1_000,
        split_seed=checkpoint["split_seed"], loader_seed=checkpoint["seed"]
    )
    embeddings, labels, _ = extract_embeddings(model, loaders.density, device)
    print("device", device, "shape", tuple(embeddings.shape), "finite", bool(torch.isfinite(embeddings).all()))
    print("min", float(embeddings.min()), "max", float(embeddings.max()), "max_abs", float(embeddings.abs().max()))
    print("zero_dimensions", int((embeddings.std(0) == 0).sum()))
    for class_index in range(10):
        values = embeddings[labels == class_index]
        centered = values - values.mean(0)
        covariance = centered.T @ centered / (len(values) - 1)
        print(
            class_index,
            "class_max_abs", float(values.abs().max()),
            "covariance_finite", bool(torch.isfinite(covariance).all()),
            "covariance_max_abs", float(torch.nan_to_num(covariance).abs().max()),
        )


if __name__ == "__main__":
    main()
