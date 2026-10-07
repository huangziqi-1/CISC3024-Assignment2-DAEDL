from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .metrics import reliability_data


def plot_reliability_diagram(
    probabilities: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
    n_bins: int = 15,
    title: str = "Pilot reliability diagram",
) -> None:
    bins = reliability_data(probabilities, labels, n_bins=n_bins)
    mask = bins.count > 0
    width = 1.0 / n_bins
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot([0, 1], [0, 1], "--", color="black", linewidth=1, label="Ideal")
    axis.bar(
        bins.mean_confidence[mask], bins.accuracy[mask], width=width * 0.8,
        alpha=0.75, edgecolor="black", label="Observed"
    )
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Accuracy", title=title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)

