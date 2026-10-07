#!/usr/bin/env python3
"""Sensitivity check: refit DAEDL GDA after removing globally constant training dimensions."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from daedl_project.artifacts import write_json
from daedl_project.data import build_cifar10_loaders, build_svhn_loader
from daedl_project.evaluation import extract_embeddings, ood_metrics_from_rows
from daedl_project.gda import density_score, fit_gda, log_density
from daedl_project.losses import LOGIT_CLAMP
from daedl_project.metrics import classification_metrics, predictive_entropy
from daedl_project.model import CifarResNet18


@torch.no_grad()
def extract_logits_embeddings(model, loader):
    logits, embeddings, labels, indices = [], [], [], []
    model.eval()
    for images, batch_labels, batch_indices in loader:
        batch_logits, batch_embeddings = model(images)
        if not torch.isfinite(batch_logits).all() or not torch.isfinite(batch_embeddings).all():
            raise FloatingPointError("non-finite output in active-dimension diagnostic")
        logits.append(batch_logits.to(dtype=torch.float64))
        embeddings.append(batch_embeddings.to(dtype=torch.float64))
        labels.append(batch_labels)
        indices.append(torch.as_tensor(batch_indices))
    return torch.cat(logits), torch.cat(embeddings), torch.cat(labels), torch.cat(indices)


def rows_from_outputs(logits, embeddings, indices, gda, labels=None):
    log_values = log_density(embeddings, gda)
    scores = density_score(log_values, gda)
    scaled = torch.clamp(logits * scores.unsqueeze(1), -LOGIT_CLAMP, LOGIT_CLAMP)
    probabilities = torch.softmax(scaled, dim=1).numpy()
    entropy = predictive_entropy(probabilities, normalized=True)
    rows = []
    for position, index in enumerate(indices.numpy()):
        row = {
            "sample_index": int(index),
            "density_score": float(scores[position]),
            "normalized_predictive_entropy": float(entropy[position]),
        }
        for class_index, value in enumerate(probabilities[position]):
            row[f"probability_{class_index}"] = float(value)
        if labels is not None:
            row["ground_truth"] = int(labels[position])
        rows.append(row)
    return probabilities, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    checkpoint = torch.load(
        args.run_dir / "best_validation_nll_checkpoint.pt", map_location="cpu", weights_only=True
    )
    if checkpoint["model_type"] != "daedl":
        raise ValueError("diagnostic requires a DAEDL run")
    model = CifarResNet18(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    loaders = build_cifar10_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        split_seed=checkpoint["split_seed"],
        loader_seed=checkpoint["seed"],
    )
    train_embeddings, train_labels, _ = extract_embeddings(model, loaders.density, torch.device("cpu"))
    standard_deviation = train_embeddings.std(dim=0, unbiased=True)
    active = standard_deviation > 1e-12
    reduced = train_embeddings[:, active]
    reduced_gda = fit_gda(reduced, train_labels, num_classes=10)

    test_logits, test_embeddings, test_labels, test_indices = extract_logits_embeddings(model, loaders.test)
    svhn_loader = build_svhn_loader(
        args.data_dir, args.batch_size, checkpoint["seed"], subset_size=10_000
    )
    svhn_logits, svhn_embeddings, _, svhn_indices = extract_logits_embeddings(model, svhn_loader)
    test_probabilities, test_rows = rows_from_outputs(
        test_logits, test_embeddings[:, active], test_indices, reduced_gda, test_labels
    )
    _, ood_rows = rows_from_outputs(svhn_logits, svhn_embeddings[:, active], svhn_indices, reduced_gda)

    primary_test = pd.read_csv(args.run_dir / "test_predictions.csv").sort_values("sample_index")
    diagnostic_test = pd.DataFrame(test_rows).sort_values("sample_index")
    primary_ood = pd.read_csv(args.run_dir / "svhn_predictions.csv").sort_values("sample_index")
    diagnostic_ood = pd.DataFrame(ood_rows).sort_values("sample_index")
    if not np.array_equal(primary_test.sample_index, diagnostic_test.sample_index):
        raise AssertionError("test index mismatch in sensitivity diagnostic")
    if not np.array_equal(primary_ood.sample_index, diagnostic_ood.sample_index):
        raise AssertionError("SVHN index mismatch in sensitivity diagnostic")

    for frame, rows in ((diagnostic_test, test_rows), (diagnostic_ood, ood_rows)):
        frame["density_uncertainty"] = 1.0 - frame.density_score
        for row, value in zip(rows, frame.density_uncertainty):
            row["density_uncertainty"] = float(value)
    primary_density = np.concatenate([primary_test.density_score, primary_ood.density_score])
    diagnostic_density = np.concatenate([diagnostic_test.density_score, diagnostic_ood.density_score])
    metrics = {
        "status": "diagnostic_only_primary_results_unchanged",
        "rule": "remove dimensions with training standard deviation <= 1e-12; no test/OOD information used",
        "original_dimensions": int(active.numel()),
        "removed_constant_dimensions": int((~active).sum()),
        "retained_dimensions": int(active.sum()),
        "retained_dimension_indices": torch.nonzero(active, as_tuple=False).flatten().tolist(),
        "reduced_gda_jitter_by_class": reduced_gda.jitter_by_class,
        "reduced_training_log_density_min": reduced_gda.log_density_min,
        "reduced_training_log_density_max": reduced_gda.log_density_max,
        "density_score_pearson_correlation_all_id_ood": float(np.corrcoef(primary_density, diagnostic_density)[0, 1]),
        "density_score_mean_absolute_change_all_id_ood": float(np.abs(primary_density - diagnostic_density).mean()),
        "clean_metrics": classification_metrics(test_probabilities, test_labels.numpy()),
        "ood_entropy": ood_metrics_from_rows(test_rows, ood_rows, "normalized_predictive_entropy"),
        "ood_density_uncertainty": ood_metrics_from_rows(test_rows, ood_rows, "density_uncertainty"),
        "id_entropy_mean": float(diagnostic_test.normalized_predictive_entropy.mean()),
        "ood_entropy_mean": float(diagnostic_ood.normalized_predictive_entropy.mean()),
    }
    write_json(args.run_dir / "density_active_dimension_sensitivity.json", metrics)
    pd.DataFrame(test_rows).sort_values("sample_index").to_csv(
        args.run_dir / "diagnostic_active_dimension_test_predictions.csv", index=False
    )
    pd.DataFrame(ood_rows).sort_values("sample_index").to_csv(
        args.run_dir / "diagnostic_active_dimension_svhn_predictions.csv", index=False
    )
    print(metrics)


if __name__ == "__main__":
    main()
