#!/usr/bin/env python3
"""Optional small SVHN smoke evaluation; never runs a full OOD experiment by default."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.daedl_project.artifacts import write_json
from src.daedl_project.data import build_cifar10_loaders, build_svhn_loader
from src.daedl_project.evaluation import evaluate_model, ood_metrics_from_rows, score_ood_loader
from src.daedl_project.gda import GDAState
from src.daedl_project.model import CifarResNet18
from src.daedl_project.reproducibility import choose_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"], default="mps")
    parser.add_argument("--subset-size", type=int, default=500)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    choice = choose_device(args.device, allow_cpu_fallback=False)
    device = torch.device(choice.effective)
    checkpoint = torch.load(args.run_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    model = CifarResNet18(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model_type = checkpoint["model_type"]

    gda_state = None
    if model_type == "daedl":
        raw = torch.load(args.run_dir / "gda_state.pt", map_location="cpu", weights_only=True)
        gda_state = GDAState(**raw)
        gda_state.validate()

    loaders = build_cifar10_loaders(
        args.data_dir,
        pilot_train_size=5_000,
        pilot_validation_size=1_000,
        pilot_test_size=args.subset_size,
        split_seed=checkpoint["split_seed"],
        loader_seed=checkpoint["seed"],
    )
    svhn = build_svhn_loader(args.data_dir, 64, checkpoint["seed"], subset_size=args.subset_size)
    _, id_rows = evaluate_model(model, loaders.test, device, model_type, gda_state)
    ood_rows = score_ood_loader(model, svhn, device, model_type, gda_state)
    metrics = ood_metrics_from_rows(id_rows, ood_rows, "normalized_predictive_entropy")
    metrics.update(
        {
            "phase": "PILOT_OOD_SMOKE_ONLY",
            "positive_class": "OOD",
            "score": "normalized_predictive_entropy",
            "higher_score_meaning": "more OOD-like",
            "id_count": len(id_rows),
            "ood_count": len(ood_rows),
        }
    )
    pd.DataFrame(id_rows).to_csv(args.run_dir / "ood_smoke_id_predictions.csv", index=False)
    pd.DataFrame(ood_rows).to_csv(args.run_dir / "ood_smoke_svhn_predictions.csv", index=False)
    write_json(args.run_dir / "ood_smoke_metrics.json", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
