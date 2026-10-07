#!/usr/bin/env python3
"""Validate and aggregate the frozen three-seed Phase 3B experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from daedl_project.artifacts import write_json
from daedl_project.evaluation import ood_metrics_from_rows
from daedl_project.metrics import classification_metrics, error_detection_metrics


MODELS = ("softmax", "daedl")
SEEDS = (3024, 3025, 3026)
METRIC_COLUMNS = (
    "accuracy", "macro_f1", "nll", "ece_15", "brier",
    "error_correct_mean_entropy", "error_correct_median_entropy",
    "error_incorrect_mean_entropy", "error_incorrect_median_entropy",
    "error_auroc", "error_aupr", "ood_id_mean_entropy", "ood_id_median_entropy",
    "ood_mean_entropy", "ood_median_entropy", "ood_auroc", "ood_aupr",
)
PAIRED_COLUMNS = (
    "accuracy", "macro_f1", "nll", "ece_15", "brier",
    "error_auroc", "ood_auroc", "ood_aupr",
)


def load_json(path: Path):
    return json.loads(path.read_text())


def probabilities(frame: pd.DataFrame) -> np.ndarray:
    return frame[[f"probability_{index}" for index in range(10)]].to_numpy(dtype=np.float64)


def check_close(name: str, actual: float, expected: float, tolerance: float = 2e-6) -> None:
    if not np.isclose(actual, expected, atol=tolerance, rtol=tolerance):
        raise AssertionError(f"{name}: recomputed={actual}, saved={expected}")


def scientific_config(config: dict) -> dict:
    excluded = {"model", "phase", "run_dir", "seed", "split_seed", "skip_individual_figures"}
    return {key: value for key, value in config.items() if key not in excluded}


def metric_row(seed: int, model: str, summary: dict) -> dict:
    test, ood = summary["test_metrics"], summary["ood_metrics"]
    error = test["error_detection"]
    return {
        "seed": seed,
        "model": model,
        "selected_epoch": summary["selected_epoch"],
        "accuracy": test["accuracy"],
        "macro_precision": test["macro_precision"],
        "macro_recall": test["macro_recall"],
        "macro_f1": test["macro_f1"],
        "nll": test["nll"],
        "ece_15": test["ece_15"],
        "brier": test["brier"],
        "error_correct_mean_entropy": error["correct_mean_uncertainty"],
        "error_correct_median_entropy": error["correct_median_uncertainty"],
        "error_incorrect_mean_entropy": error["incorrect_mean_uncertainty"],
        "error_incorrect_median_entropy": error["incorrect_median_uncertainty"],
        "error_auroc": error["auroc"],
        "error_aupr": error["aupr"],
        "ood_id_mean_entropy": ood["id_uncertainty"]["mean"],
        "ood_id_median_entropy": ood["id_uncertainty"]["median"],
        "ood_mean_entropy": ood["ood_uncertainty"]["mean"],
        "ood_median_entropy": ood["ood_uncertainty"]["median"],
        "ood_auroc": ood["auroc"],
        "ood_aupr": ood["aupr"],
    }


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    shown = frame.copy()
    for column in shown.select_dtypes(include=[np.number]).columns:
        if pd.api.types.is_integer_dtype(shown[column]):
            shown[column] = shown[column].map(lambda value: str(int(value)))
        else:
            shown[column] = shown[column].map(
                lambda value: f"{value:.2e}" if value != 0 and abs(value) < 10 ** (-digits) else f"{value:.{digits}f}"
            )
    header = "| " + " | ".join(map(str, shown.columns)) + " |"
    divider = "| " + " | ".join("---" for _ in shown.columns) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in shown.to_numpy()]
    return "\n".join([header, divider, *rows])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase3b_three_seed_analysis"))
    parser.add_argument("--report", type=Path, default=Path("PHASE3B_RESULTS.md"))
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    directories = {
        (3024, "softmax"): args.project_root / "results/final_core_softmax_seed3024_phase3a",
        (3024, "daedl"): args.project_root / "results/final_core_daedl_seed3024_phase3a",
        **{
            (seed, model): args.project_root / f"results/final_replication_{model}_seed{seed}"
            for seed in (3025, 3026) for model in MODELS
        },
    }
    checks: dict[str, object] = {"status": "PASS", "runs": {}}
    metric_rows, density_rows = [], []
    seed3024_frames: dict[str, pd.DataFrame] = {}
    frozen_reference = None
    cross_seed_splits = []

    for seed in SEEDS:
        within_seed_split = within_seed_ood = None
        for model in MODELS:
            run_dir = directories[(seed, model)]
            expected_phase = "FINAL_CORE" if seed == 3024 else "FINAL_REPLICATION"
            for required in (
                expected_phase, "COMPLETED", "config.json", "environment.json", "split_indices.npz",
                "training_history.csv", "best_validation_nll_checkpoint.pt", "final_checkpoint.pt",
                "test_predictions.csv", "svhn_predictions.csv", "test_metrics.json", "ood_metrics.json",
                "run_summary.json",
            ):
                if not (run_dir / required).exists():
                    raise AssertionError(f"seed {seed} {model} missing {required}")

            config = load_json(run_dir / "config.json")
            if config["phase"] != expected_phase or config["seed"] != seed or config["split_seed"] != seed:
                raise AssertionError(f"seed/phase marker mismatch for {run_dir}")
            if config["epochs"] != 30 or config["checkpoint_selection"] != "minimum validation NLL; test set never consulted":
                raise AssertionError(f"frozen training protocol mismatch for {run_dir}")
            current_science = scientific_config(config)
            if frozen_reference is None:
                frozen_reference = current_science
            elif current_science != frozen_reference:
                raise AssertionError(f"scientific configuration changed for {run_dir}")

            split_file = np.load(run_dir / "split_indices.npz")
            split = tuple(split_file[key] for key in ("train", "validation", "test"))
            if tuple(map(len, split)) != (45_000, 5_000, 10_000):
                raise AssertionError(f"split sizes mismatch for {run_dir}")
            if np.intersect1d(split[0], split[1]).size:
                raise AssertionError(f"train/validation overlap for {run_dir}")
            if within_seed_split is None:
                within_seed_split = split
                cross_seed_splits.append(split[1].copy())
            elif not all(np.array_equal(left, right) for left, right in zip(within_seed_split, split)):
                raise AssertionError(f"models do not share split at seed {seed}")

            test_frame = pd.read_csv(run_dir / "test_predictions.csv").sort_values("sample_index")
            ood_frame = pd.read_csv(run_dir / "svhn_predictions.csv").sort_values("sample_index")
            for label, frame in (("test", test_frame), ("SVHN", ood_frame)):
                numeric = frame.select_dtypes(include=[np.number]).to_numpy()
                if not np.isfinite(numeric).all():
                    raise AssertionError(f"{run_dir} {label} contains NaN/Inf")
                if np.abs(probabilities(frame).sum(axis=1) - 1.0).max() > 1e-5:
                    raise AssertionError(f"{run_dir} {label} probabilities are not normalized")
            if not np.array_equal(test_frame.sample_index.to_numpy(), np.arange(10_000)):
                raise AssertionError(f"unexpected CIFAR test indices for {run_dir}")
            if within_seed_ood is None:
                within_seed_ood = ood_frame.sample_index.to_numpy()
            elif not np.array_equal(within_seed_ood, ood_frame.sample_index.to_numpy()):
                raise AssertionError(f"models do not share SVHN subset at seed {seed}")
            if model == "daedl" and np.any(test_frame[[f"alpha_{i}" for i in range(10)]].to_numpy() <= 0):
                raise AssertionError(f"non-positive DAEDL concentration at seed {seed}")

            saved_test = load_json(run_dir / "test_metrics.json")
            recomputed_test = classification_metrics(probabilities(test_frame), test_frame.ground_truth.to_numpy())
            for key, value in recomputed_test.items():
                check_close(f"seed {seed} {model} {key}", value, saved_test[key])
            recomputed_error = error_detection_metrics(
                test_frame.correct.astype(bool).to_numpy(), test_frame.normalized_predictive_entropy.to_numpy()
            )
            for key in (
                "auroc", "aupr", "correct_mean_uncertainty", "correct_median_uncertainty",
                "incorrect_mean_uncertainty", "incorrect_median_uncertainty",
            ):
                check_close(f"seed {seed} {model} error {key}", recomputed_error[key], saved_test["error_detection"][key])
            saved_ood = load_json(run_dir / "ood_metrics.json")
            if (saved_ood["positive_class"] != "OOD" or saved_ood["higher_score_meaning"] != "more OOD-like"
                    or saved_ood["id_count"] != 10_000 or saved_ood["ood_count"] != 10_000):
                raise AssertionError(f"ID/OOD label or score direction mismatch for {run_dir}")
            recomputed_ood = ood_metrics_from_rows(
                test_frame.to_dict("records"), ood_frame.to_dict("records"), "normalized_predictive_entropy"
            )
            for key in ("auroc", "aupr"):
                check_close(f"seed {seed} {model} OOD {key}", recomputed_ood[key], saved_ood[key])

            history = pd.read_csv(run_dir / "training_history.csv")
            selected = history.loc[history.validation_nll.idxmin()]
            checkpoint = torch.load(run_dir / "best_validation_nll_checkpoint.pt", map_location="cpu", weights_only=True)
            final_checkpoint = torch.load(run_dir / "final_checkpoint.pt", map_location="cpu", weights_only=True)
            if checkpoint["selected_epoch"] != int(selected.epoch) or final_checkpoint["epoch"] != 30:
                raise AssertionError(f"checkpoint mismatch for {run_dir}")
            if checkpoint.get("seed") != seed or final_checkpoint.get("seed") != seed:
                raise AssertionError(f"checkpoint seed mismatch for {run_dir}")
            check_close("checkpoint validation NLL", float(selected.validation_nll), checkpoint["validation_nll"], 1e-8)

            summary = load_json(run_dir / "run_summary.json")
            metric_rows.append(metric_row(seed, model, summary))
            if seed == 3024:
                seed3024_frames[model] = test_frame
                seed3024_frames[f"{model}_ood"] = ood_frame
            run_check = {
                "directory": str(run_dir), "phase": expected_phase, "all_numeric_outputs_finite": True,
                "metrics_recomputed_match": True, "probabilities_normalized": True,
                "selected_epoch": int(selected.epoch), "selected_checkpoint_is_minimum_validation_nll": True,
                "checkpoint_round_trip_verified_during_run": saved_test["checkpoint_round_trip_verified"],
            }
            if model == "daedl":
                diagnostic = load_json(run_dir / "density_diagnostics.json")
                secondary = saved_ood["secondary_density_uncertainty"]
                ranks = [item["covariance_rank"] for item in diagnostic["classes"]]
                jitter = diagnostic["jitter_by_class"]
                density_rows.append({
                    "seed": seed, "global_embedding_rank": diagnostic["global_embedding_rank"],
                    "constant_dimensions": diagnostic["constant_dimensions"],
                    "near_constant_dimensions": diagnostic["near_constant_dimensions"],
                    "class_rank_min": min(ranks), "class_rank_max": max(ranks),
                    "jitter_min": min(jitter), "jitter_max": max(jitter),
                    "training_log_density_min": diagnostic["training_log_density_min"],
                    "training_log_density_max": diagnostic["training_log_density_max"],
                    "nonfinite_fraction": diagnostic["nonfinite_fraction"],
                    "density_ood_auroc": secondary["auroc"], "density_ood_aupr": secondary["aupr"],
                })
                if diagnostic["nonfinite_fraction"] != 0:
                    raise AssertionError(f"non-finite DAEDL density at seed {seed}")
            checks["runs"][f"{model}_seed{seed}"] = run_check

    if any(np.array_equal(cross_seed_splits[0], other) for other in cross_seed_splits[1:]):
        raise AssertionError("validation split did not vary across training seeds")
    checks.update({
        "frozen_scientific_configuration_across_all_six_runs": True,
        "identical_split_indices_within_each_seed": True,
        "split_varies_across_seeds": True,
        "identical_svhn_indices_within_each_seed": True,
        "test_indices_identical_across_all_runs": True,
        "training_seed_is_replication_unit": True,
        "aggregate_sd_definition": "sample standard deviation across seeds (ddof=1)",
        "phase3a_seed3024_artifacts_reused_without_rerun": True,
    })

    per_seed = pd.DataFrame(metric_rows)
    per_seed["model"] = pd.Categorical(per_seed.model, categories=MODELS, ordered=True)
    per_seed = per_seed.sort_values(["seed", "model"])
    per_seed.to_csv(args.output_dir / "per_seed_metrics.csv", index=False)
    aggregate_rows = []
    for model, group in per_seed.groupby("model", sort=False):
        for metric in METRIC_COLUMNS:
            aggregate_rows.append({
                "model": model, "metric": metric, "mean": group[metric].mean(),
                "sample_sd": group[metric].std(ddof=1), "n_seeds": len(group),
            })
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(args.output_dir / "aggregate_metrics.csv", index=False)

    paired_rows = []
    indexed = per_seed.set_index(["seed", "model"])
    for seed in SEEDS:
        for metric in PAIRED_COLUMNS:
            paired_rows.append({
                "seed": seed, "metric": metric,
                "daedl_minus_softmax": indexed.loc[(seed, "daedl"), metric] - indexed.loc[(seed, "softmax"), metric],
            })
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(args.output_dir / "paired_differences.csv", index=False)
    density = pd.DataFrame(density_rows)
    density.to_csv(args.output_dir / "density_diagnostics_by_seed.csv", index=False)
    write_json(args.output_dir / "consistency_checks.json", checks)

    # Figure A: seed points plus three-seed means for the four headline metrics.
    figure, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    display_metrics = (("accuracy", "Clean accuracy"), ("ece_15", "ECE (15 bins)"),
                       ("error_auroc", "Error-detection AUROC"), ("ood_auroc", "SVHN OOD AUROC"))
    colors = {"softmax": "#4C78A8", "daedl": "#F58518"}
    for axis, (metric, title) in zip(axes.flat, display_metrics):
        for position, model in enumerate(MODELS):
            values = per_seed.loc[per_seed.model == model, metric].to_numpy()
            axis.scatter(np.full(3, position), values, color=colors[model], s=38, alpha=0.75, zorder=2)
            axis.errorbar(position, values.mean(), yerr=values.std(ddof=1), fmt="_", markersize=20,
                          capsize=5, linewidth=2, color="black", zorder=3)
        axis.set_xticks(range(2), ("Softmax", "DAEDL"))
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Frozen three-seed results: points are seeds; black bars are mean ± sample SD")
    figure.tight_layout()
    figure.savefig(args.output_dir / "three_seed_headline_metrics.png", dpi=180)
    plt.close(figure)

    # Figure B: the accepted seed-3024 distributions, reused rather than regenerating six redundant panels.
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for column, model in enumerate(MODELS):
        clean = seed3024_frames[model]
        for correct, label in ((True, "correct"), (False, "incorrect")):
            axes[0, column].hist(clean.loc[clean.correct.astype(bool) == correct, "normalized_predictive_entropy"],
                                 bins=40, density=True, alpha=0.45, label=label)
        for frame, label in ((clean, "CIFAR-10"), (seed3024_frames[f"{model}_ood"], "SVHN")):
            axes[1, column].hist(frame.normalized_predictive_entropy, bins=40, density=True, alpha=0.45, label=label)
        axes[0, column].set_title(f"{model}: errors (seed 3024)")
        axes[1, column].set_title(f"{model}: OOD (seed 3024)")
        for axis in axes[:, column]:
            axis.set_xlabel("Normalized predictive entropy")
            axis.set_ylabel("Density")
            axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "seed3024_entropy_distributions.png", dpi=180)
    plt.close(figure)

    # Generate the required report tables from the machine-readable source files.
    class_columns = ["seed", "model", "accuracy", "macro_precision", "macro_recall", "macro_f1"]
    calibration_columns = ["seed", "model", "nll", "ece_15", "brier"]
    error_columns = ["seed", "model", "error_correct_mean_entropy", "error_correct_median_entropy",
                     "error_incorrect_mean_entropy", "error_incorrect_median_entropy", "error_auroc", "error_aupr"]
    ood_columns = ["seed", "model", "ood_id_mean_entropy", "ood_id_median_entropy", "ood_mean_entropy",
                   "ood_median_entropy", "ood_auroc", "ood_aupr"]
    aggregate_wide = aggregate.copy()
    aggregate_wide["mean ± sample SD"] = aggregate_wide.apply(
        lambda row: f"{row['mean']:.4f} ± {row['sample_sd']:.4f}", axis=1
    )
    aggregate_wide = aggregate_wide.pivot(index="metric", columns="model", values="mean ± sample SD").reset_index()
    aggregate_wide = aggregate_wide[["metric", "softmax", "daedl"]]
    paired_wide = paired.pivot(index="metric", columns="seed", values="daedl_minus_softmax").reset_index()
    report = f"""# Phase 3B: Frozen three-seed replication

## Protocol

Seeds 3024, 3025, and 3026 are independent training/split replicates. The Phase-3A seed-3024 artifacts were reused unchanged. Seeds 3025 and 3026 used the frozen Phase-3A architecture, optimization, validation-NLL checkpoint rule, DAEDL objective, density estimator, and evaluation definitions. Values below are generated from saved per-example predictions; aggregate variability is the sample standard deviation across the three seeds (`ddof=1`). OOD is SVHN and the positive class is OOD.

## Per-seed clean classification and calibration

{markdown_table(per_seed[class_columns])}

## Per-seed calibration

{markdown_table(per_seed[calibration_columns])}

## Per-seed error uncertainty

{markdown_table(per_seed[error_columns])}

## Per-seed OOD uncertainty

{markdown_table(per_seed[ood_columns])}

## Three-seed aggregate

{markdown_table(aggregate_wide, digits=4)}

## Paired differences (DAEDL minus Softmax)

Positive values favor DAEDL only for accuracy, macro F1, error-detection AUROC, and OOD AUROC/AUPR. Negative values favor DAEDL for NLL, ECE, and Brier.

{markdown_table(paired_wide)}

## DAEDL density diagnostics

{markdown_table(density)}

## Consistency

- Clean accuracy and macro F1 consistently favor Softmax across seeds. DAEDL's ECE is consistently lower, while Softmax consistently has lower NLL and Brier score.
- Both methods consistently assign higher mean and median entropy to incorrect than correct predictions. Error-detection AUROC favors Softmax in all three seeds, whereas error-detection AUPR has mixed paired signs.
- DAEDL consistently assigns greater mean and median entropy to SVHN than CIFAR-10.

## Non-replicated findings

- Softmax does not outperform DAEDL on SVHN entropy AUROC in every seed: DAEDL is higher at seed 3025, while Softmax is higher at seeds 3024 and 3026. The report therefore does not claim a stable method-level OOD advantage from this benchmark.
- Error-detection AUPR and SVHN OOD AUPR also change paired direction across seeds.

## DAEDL density stability

- Rank deficiency persists but varies: global rank rises from 93 to 99 and 101, while constant dimensions fall from 35 to 29 and 27. Per-class ranks span 76–95 across the three seeds.
- Density fitting is numerically robust: all non-finite fractions are zero and required adaptive jitter remains of order `1e-8` to `1e-7`.
- The secondary density score improves over DAEDL entropy only at seed 3024; it is worse at seeds 3025 and 3026, so the improvement is not stable.

## Interpretation

The results support a narrow trade-off: DAEDL consistently improves ECE but not clean predictive quality under NLL/Brier, and it does not provide a stable SVHN OOD advantage. No validity failure was found: all six runs use the frozen scientific configuration, within-seed comparisons share data indices, ID/OOD labels and score direction are correct, checkpoint selection uses validation NLL only, and all recomputed metrics match saved values. With only three seeds, these are descriptive consistency findings rather than significance claims.

## Figures and machine-readable sources

- `results/phase3b_three_seed_analysis/three_seed_headline_metrics.png`
- `results/phase3b_three_seed_analysis/seed3024_entropy_distributions.png`
- `results/phase3b_three_seed_analysis/per_seed_metrics.csv`
- `results/phase3b_three_seed_analysis/aggregate_metrics.csv`
- `results/phase3b_three_seed_analysis/paired_differences.csv`
- `results/phase3b_three_seed_analysis/density_diagnostics_by_seed.csv`
- `results/phase3b_three_seed_analysis/consistency_checks.json`

## Recommendation

**Add small corruption experiment.** The three-seed clean/calibration pattern is stable, but SVHN OOD rankings vary by seed. A compact, pre-specified corruption evaluation would test uncertainty under a complementary covariate shift without tuning or rerunning these models.
"""
    args.report.write_text(report)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
