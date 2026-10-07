#!/usr/bin/env python3
"""Build the compact, tracked evidence package from validated final artifacts."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results/phase3b_three_seed_analysis"
SUMMARY = ROOT / "results/final_summary"
FIGURES = ROOT / "figures/final"


def load_json(path: Path):
    return json.loads(path.read_text())


def validate_metrics() -> None:
    aggregate = pd.read_csv(ANALYSIS / "aggregate_metrics.csv").set_index(["model", "metric"])
    expected = {
        ("softmax", "accuracy"): (0.9013333333333334, 0.00040414518843275764),
        ("daedl", "accuracy"): (0.8902, 0.0012489995996796488),
        ("softmax", "ece_15"): (0.04629426133334636, 0.0031440793139860443),
        ("daedl", "ece_15"): (0.026135083961486814, 0.0029989186310392813),
        ("softmax", "ood_auroc"): (0.8860282833333333, 0.013976173040296313),
        ("daedl", "ood_auroc"): (0.8761567533333334, 0.023789612627145858),
    }
    for key, (mean, sd) in expected.items():
        row = aggregate.loc[key]
        if not np.isclose(row["mean"], mean) or not np.isclose(row["sample_sd"], sd):
            raise AssertionError(f"authoritative aggregate mismatch for {key}")
    checks = load_json(ANALYSIS / "consistency_checks.json")
    if checks.get("status") != "PASS":
        raise AssertionError("Phase 3B consistency audit did not pass")


def prepare_configuration() -> None:
    source = load_json(ROOT / "results/final_core_daedl_seed3024_phase3a/config.json")
    for key in ("run_dir", "phase", "seed", "split_seed", "model", "skip_individual_figures"):
        source.pop(key, None)
    source.update(
        {
            "experiment_name": "frozen_three_seed_softmax_daedl_comparison",
            "models": ["softmax", "daedl"],
            "seeds": [3024, 3025, 3026],
            "replicate_unit": "training seed / model run",
            "aggregate_variability": "sample standard deviation (ddof=1)",
            "ood_dataset": "SVHN test split",
            "primary_common_uncertainty_score": "normalized predictive entropy",
        }
    )
    (ROOT / "configs").mkdir(exist_ok=True)
    (ROOT / "configs/final_experiment.json").write_text(json.dumps(source, indent=2, sort_keys=True) + "\n")


def copy_evidence() -> None:
    SUMMARY.mkdir(parents=True, exist_ok=True)
    for name in (
        "per_seed_metrics.csv", "aggregate_metrics.csv", "paired_differences.csv",
        "density_diagnostics_by_seed.csv", "consistency_checks.json",
    ):
        shutil.copy2(ANALYSIS / name, SUMMARY / name)
    sources = {
        "reliability_softmax_seed3024.csv": ROOT / "results/final_core_softmax_seed3024_phase3a/figure_input_reliability.csv",
        "reliability_daedl_seed3024.csv": ROOT / "results/final_core_daedl_seed3024_phase3a/figure_input_reliability.csv",
        "error_entropy_softmax_seed3024.csv": ROOT / "results/final_core_softmax_seed3024_phase3a/figure_input_error_entropy.csv",
        "error_entropy_daedl_seed3024.csv": ROOT / "results/final_core_daedl_seed3024_phase3a/figure_input_error_entropy.csv",
        "ood_entropy_softmax_seed3024.csv": ROOT / "results/final_core_softmax_seed3024_phase3a/figure_input_ood_entropy.csv",
        "ood_entropy_daedl_seed3024.csv": ROOT / "results/final_core_daedl_seed3024_phase3a/figure_input_ood_entropy.csv",
    }
    for output, source in sources.items():
        shutil.copy2(source, SUMMARY / output)


def conceptual_figure(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(12.5, 4.4))
    axis.set_xlim(0, 12.5)
    axis.set_ylim(0, 4.4)
    axis.axis("off")
    navy, pale, orange, gray = "#17365D", "#EAF1F8", "#D97904", "#5B6573"

    def box(x, y, width, height, title, detail, fill=pale, edge=navy):
        patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.06,rounding_size=0.08",
                               facecolor=fill, edgecolor=edge, linewidth=1.5)
        axis.add_patch(patch)
        axis.text(x + width / 2, y + height * 0.64, title, ha="center", va="center",
                  fontsize=11, fontweight="bold", color="#111111")
        axis.text(x + width / 2, y + height * 0.32, detail, ha="center", va="center",
                  fontsize=8.5, color="#333333", wrap=True)

    def arrow(x1, y1, x2, y2, label=None):
        axis.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                       linewidth=1.4, color=gray))
        if label:
            axis.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, label, ha="center", fontsize=8, color=gray)

    box(0.25, 1.45, 1.45, 1.3, "Input image", "CIFAR-10 or\nunseen image", fill="#F4F4F4", edge=gray)
    box(2.15, 1.45, 2.0, 1.3, "Feature extractor", "Spectrally normalized\nCIFAR ResNet-18")
    box(4.75, 2.65, 1.65, 1.15, "Classifier", "Logits z(x)")
    box(4.75, 0.55, 1.65, 1.15, "Feature density", "GDA on 128-D\nembedding", fill="#FFF4E6", edge=orange)
    box(7.0, 0.95, 2.15, 2.0, "Density-aware evidence", "Scale logits by s(x)\nα = exp(s(x) z(x))",
        fill="#FFF4E6", edge=orange)
    box(9.8, 1.45, 2.35, 1.3, "Prediction", "Dirichlet mean probabilities\nand predictive entropy")
    arrow(1.7, 2.1, 2.15, 2.1)
    arrow(4.15, 2.25, 4.75, 3.1, "logits")
    arrow(4.15, 1.95, 4.75, 1.12, "embedding")
    arrow(6.4, 3.22, 7.0, 2.45)
    arrow(6.4, 1.12, 7.0, 1.45, "s(x)")
    arrow(9.15, 1.95, 9.8, 2.1)
    axis.text(6.25, 4.18, "Training learns features and logits; inference adds feature density",
              ha="center", va="center", fontsize=12, fontweight="bold", color="#111111")
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def reliability_figure(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    axis.plot([0, 1], [0, 1], "--", color="#333333", linewidth=1.2, label="Ideal")
    for model, color in (("softmax", "#4C78A8"), ("daedl", "#F58518")):
        data = pd.read_csv(SUMMARY / f"reliability_{model}_seed3024.csv")
        shown = data[data["count"] >= 50]
        axis.plot(shown.mean_confidence, shown.accuracy, "o-", color=color, linewidth=1.8,
                  markersize=5.5, label=model.capitalize())
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Accuracy",
             title="CIFAR-10 reliability at seed 3024")
    axis.text(0.02, -0.15, "Equal-width bins with fewer than 50 test examples are suppressed.",
              transform=axis.transAxes, fontsize=8, color="#555555")
    axis.grid(alpha=0.18)
    axis.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def prepare_figures() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    conceptual_figure(FIGURES / "figure1_daedl_concept.png")
    shutil.copy2(ANALYSIS / "three_seed_headline_metrics.png", FIGURES / "figure2_three_seed_headline.png")
    reliability_figure(FIGURES / "figure3_reliability_seed3024.png")
    shutil.copy2(ANALYSIS / "seed3024_entropy_distributions.png", FIGURES / "figure4_entropy_distributions_seed3024.png")


def main() -> None:
    validate_metrics()
    prepare_configuration()
    copy_evidence()
    prepare_figures()
    print(json.dumps({"status": "PASS", "summary": str(SUMMARY), "figures": str(FIGURES)}, indent=2))


if __name__ == "__main__":
    main()
