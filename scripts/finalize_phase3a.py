#!/usr/bin/env python3
"""Cross-run validation and combined analysis figures for Phase 3A."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from torchvision import datasets

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from daedl_project.artifacts import write_json
from daedl_project.evaluation import ood_metrics_from_rows
from daedl_project.metrics import classification_metrics, error_detection_metrics, reliability_data


MODELS = ("softmax", "daedl")


def load_json(path: Path):
    return json.loads(path.read_text())


def probabilities(frame: pd.DataFrame) -> np.ndarray:
    return frame[[f"probability_{index}" for index in range(10)]].to_numpy(dtype=np.float64)


def check_close(name: str, actual: float, expected: float, tolerance: float = 2e-6) -> None:
    if not np.isclose(actual, expected, atol=tolerance, rtol=tolerance):
        raise AssertionError(f"{name}: recomputed={actual}, saved={expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--softmax-dir", type=Path, required=True)
    parser.add_argument("--daedl-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    run_dirs = {"softmax": args.softmax_dir, "daedl": args.daedl_dir}
    checks: dict[str, object] = {"status": "PASS", "models": {}}
    frames, ood_frames, histories, metrics, configs = {}, {}, {}, {}, {}

    split_reference = None
    test_reference = None
    ood_reference = None
    for model_name, run_dir in run_dirs.items():
        for required in ("FINAL_CORE", "COMPLETED", "best_validation_nll_checkpoint.pt", "final_checkpoint.pt"):
            if not (run_dir / required).exists():
                raise AssertionError(f"{model_name} missing {required}")
        split = np.load(run_dir / "split_indices.npz")
        if (len(split["train"]), len(split["validation"]), len(split["test"])) != (45_000, 5_000, 10_000):
            raise AssertionError(f"{model_name} split sizes are wrong")
        if np.intersect1d(split["train"], split["validation"]).size:
            raise AssertionError(f"{model_name} train/validation overlap")
        split_tuple = tuple(split[key] for key in ("train", "validation", "test"))
        if split_reference is None:
            split_reference = split_tuple
        elif not all(np.array_equal(left, right) for left, right in zip(split_reference, split_tuple)):
            raise AssertionError("models do not share identical splits")

        frame = pd.read_csv(run_dir / "test_predictions.csv").sort_values("sample_index")
        ood = pd.read_csv(run_dir / "svhn_predictions.csv").sort_values("sample_index")
        for label, value in (("test", frame), ("SVHN", ood)):
            numeric = value.select_dtypes(include=[np.number]).to_numpy()
            if not np.isfinite(numeric).all():
                raise AssertionError(f"{model_name} {label} output contains NaN/Inf")
            maximum_probability_error = float(np.abs(probabilities(value).sum(axis=1) - 1.0).max())
            if maximum_probability_error > 1e-5:
                raise AssertionError(f"{model_name} {label} probability normalization failed")
        if model_name == "daedl":
            alpha = frame[[f"alpha_{index}" for index in range(10)]].to_numpy()
            if np.any(alpha <= 0):
                raise AssertionError("DAEDL has non-positive concentration")
        if test_reference is None:
            test_reference = frame.sample_index.to_numpy()
            ood_reference = ood.sample_index.to_numpy()
        elif not np.array_equal(test_reference, frame.sample_index.to_numpy()):
            raise AssertionError("test indices differ between models")
        elif not np.array_equal(ood_reference, ood.sample_index.to_numpy()):
            raise AssertionError("SVHN indices differ between models")

        saved = load_json(run_dir / "test_metrics.json")
        recomputed = classification_metrics(probabilities(frame), frame.ground_truth.to_numpy())
        for key, value in recomputed.items():
            check_close(f"{model_name} {key}", value, saved[key])
        error = error_detection_metrics(
            frame.correct.astype(bool).to_numpy(), frame.normalized_predictive_entropy.to_numpy()
        )
        for key in ("auroc", "aupr", "correct_mean_uncertainty", "correct_median_uncertainty",
                    "incorrect_mean_uncertainty", "incorrect_median_uncertainty"):
            check_close(f"{model_name} error {key}", error[key], saved["error_detection"][key])
        saved_ood = load_json(run_dir / "ood_metrics.json")
        recomputed_ood = ood_metrics_from_rows(
            frame.to_dict("records"), ood.to_dict("records"), "normalized_predictive_entropy"
        )
        for key in ("auroc", "aupr"):
            check_close(f"{model_name} OOD {key}", recomputed_ood[key], saved_ood[key])

        history = pd.read_csv(run_dir / "training_history.csv")
        selected_row = history.loc[history.validation_nll.idxmin()]
        checkpoint = torch.load(run_dir / "best_validation_nll_checkpoint.pt", map_location="cpu", weights_only=True)
        final_checkpoint = torch.load(run_dir / "final_checkpoint.pt", map_location="cpu", weights_only=True)
        if int(selected_row.epoch) != checkpoint["selected_epoch"] or final_checkpoint["epoch"] != 30:
            raise AssertionError(f"{model_name} checkpoint selection mismatch")
        check_close(
            f"{model_name} selected validation NLL",
            float(selected_row.validation_nll),
            checkpoint["validation_nll"],
            tolerance=1e-8,
        )
        config = load_json(run_dir / "config.json")
        if config["checkpoint_selection"] != "minimum validation NLL; test set never consulted":
            raise AssertionError("checkpoint criterion documentation mismatch")
        if config["phase"] != "FINAL_CORE" or not config["spectral_normalization"]["enabled"]:
            raise AssertionError("FINAL_CORE or spectral-normalization marker missing")
        checks["models"][model_name] = {
            "probability_max_sum_error_test": float(np.abs(probabilities(frame).sum(axis=1) - 1.0).max()),
            "probability_max_sum_error_svhn": float(np.abs(probabilities(ood).sum(axis=1) - 1.0).max()),
            "all_numeric_outputs_finite": True,
            "metrics_recomputed_match": True,
            "selected_epoch": int(selected_row.epoch),
            "selected_checkpoint_is_minimum_validation_nll": True,
            "checkpoint_round_trip_verified_during_run": saved["checkpoint_round_trip_verified"],
        }
        frames[model_name], ood_frames[model_name] = frame, ood
        histories[model_name], metrics[model_name], configs[model_name] = history, saved, config

    if configs["softmax"]["model_kwargs"] != configs["daedl"]["model_kwargs"]:
        raise AssertionError("model backbone configuration differs")
    checks.update(
        {
            "identical_train_validation_test_indices": True,
            "identical_svhn_indices": True,
            "train_validation_overlap": 0,
            "shared_backbone_and_spectral_normalization": True,
            "test_not_used_for_checkpoint_selection": True,
            "ood_positive_score_direction": "higher normalized predictive entropy means more OOD-like",
            "daedl_density_nonfinite_fraction": load_json(args.daedl_dir / "density_diagnostics.json")["nonfinite_fraction"],
            "active_dimension_sensitivity_completed": (args.daedl_dir / "density_active_dimension_sensitivity.json").exists(),
        }
    )
    write_json(args.output_dir / "consistency_checks.json", checks)

    table_rows = []
    for model_name in MODELS:
        summary = load_json(run_dirs[model_name] / "run_summary.json")
        test = summary["test_metrics"]
        ood = summary["ood_metrics"]
        table_rows.append(
            {
                "model": model_name,
                "selected_epoch": summary["selected_epoch"],
                "accuracy": test["accuracy"],
                "macro_f1": test["macro_f1"],
                "nll": test["nll"],
                "ece_15": test["ece_15"],
                "brier": test["brier"],
                "error_auroc": test["error_detection"]["auroc"],
                "error_aupr": test["error_detection"]["aupr"],
                "ood_auroc": ood["auroc"],
                "ood_aupr": ood["aupr"],
            }
        )
    pd.DataFrame(table_rows).to_csv(args.output_dir / "comparison_metrics.csv", index=False)

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for model_name in MODELS:
        history = histories[model_name]
        axes[0, 0].plot(history.epoch, history.train_accuracy, label=model_name)
        axes[0, 1].plot(history.epoch, history.validation_accuracy, label=model_name)
        axes[1, 0].plot(history.epoch, history.validation_nll, label=model_name)
        axes[1, 1].plot(history.epoch, history.learning_rate, label=model_name)
    for axis, title, ylabel in zip(
        axes.flat,
        ("Training accuracy", "Validation accuracy", "Validation NLL", "Cosine learning rate"),
        ("Accuracy", "Accuracy", "NLL", "Learning rate"),
    ):
        axis.set(title=title, xlabel="Epoch", ylabel=ylabel)
        axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "combined_training_validation_curves.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot([0, 1], [0, 1], "--", color="black", linewidth=1, label="Ideal")
    for model_name in MODELS:
        bins = reliability_data(probabilities(frames[model_name]), frames[model_name].ground_truth, 15)
        mask = bins.count > 0
        axis.plot(bins.mean_confidence[mask], bins.accuracy[mask], "o-", label=model_name)
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Accuracy", title="CIFAR-10 reliability")
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "combined_reliability_diagram.png", dpi=180)
    plt.close(figure)

    for kind, source_frames, output, title, groups in (
        ("error", frames, "combined_error_entropy_distribution.png", "Correct vs incorrect entropy", (True, False)),
        ("ood", None, "combined_ood_entropy_distribution.png", "CIFAR-10 vs SVHN entropy", ("CIFAR-10", "SVHN")),
    ):
        figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
        for axis, model_name in zip(axes, MODELS):
            if kind == "error":
                for correct, label in zip(groups, ("correct", "incorrect")):
                    values = frames[model_name].loc[
                        frames[model_name].correct.astype(bool) == correct, "normalized_predictive_entropy"
                    ]
                    axis.hist(values, bins=40, density=True, alpha=0.45, label=label)
            else:
                for data, label in ((frames[model_name], "CIFAR-10"), (ood_frames[model_name], "SVHN")):
                    axis.hist(data.normalized_predictive_entropy, bins=40, density=True, alpha=0.45, label=label)
            axis.set(title=model_name, xlabel="Normalized predictive entropy", ylabel="Density")
            axis.legend()
        figure.suptitle(title)
        figure.tight_layout()
        figure.savefig(args.output_dir / output, dpi=180)
        plt.close(figure)

    cifar = datasets.CIFAR10(args.data_dir, train=False, download=False)
    svhn = datasets.SVHN(args.data_dir, split="test", download=False)
    figure, axes = plt.subplots(2, 5, figsize=(15, 7.5))
    combined_selection = {}
    for row_axis, model_name in zip(axes, MODELS):
        selected = load_json(run_dirs[model_name] / "representative_selection.json")
        combined_selection[model_name] = selected
        for axis, item in zip(row_axis, selected):
            image, _ = (cifar if item["dataset"] == "CIFAR-10" else svhn)[int(item["sample_index"])]
            axis.imshow(image)
            axis.axis("off")
            axis.set_title(
                f"{model_name}: {item['rule']}\nidx={item['sample_index']}, pred={item['predicted_label']}\nH={item['normalized_predictive_entropy']:.3f}",
                fontsize=7,
            )
    figure.subplots_adjust(hspace=0.42, wspace=0.05, top=0.94, bottom=0.04)
    figure.savefig(args.output_dir / "combined_representative_prediction_grid.png", dpi=180)
    plt.close(figure)
    write_json(args.output_dir / "representative_selection.json", combined_selection)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
