from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


EPS = 1e-12


def validate_probabilities(probabilities: np.ndarray, atol: float = 1e-5) -> None:
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [samples, classes]")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain NaN or infinity")
    if np.any(probabilities < -atol) or np.any(probabilities > 1.0 + atol):
        raise ValueError("probabilities fall outside [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=atol):
        raise ValueError("probability rows do not sum to one")


def negative_log_likelihood(probabilities: np.ndarray, labels: np.ndarray) -> float:
    validate_probabilities(probabilities)
    labels = np.asarray(labels, dtype=np.int64)
    selected = probabilities[np.arange(labels.size), labels]
    return float(-np.log(np.clip(selected, EPS, 1.0)).mean())


def multiclass_brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    validate_probabilities(probabilities)
    labels = np.asarray(labels, dtype=np.int64)
    targets = np.eye(probabilities.shape[1], dtype=np.float64)[labels]
    return float(np.square(probabilities - targets).sum(axis=1).mean())


@dataclass(frozen=True)
class ReliabilityData:
    lower: np.ndarray
    upper: np.ndarray
    count: np.ndarray
    mean_confidence: np.ndarray
    accuracy: np.ndarray


def reliability_data(probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> ReliabilityData:
    validate_probabilities(probabilities)
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    labels = np.asarray(labels, dtype=np.int64)
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels
    # Equal-width bins: [0,1/n), ..., [(n-1)/n,1], so confidence 1 stays in the last bin.
    bin_index = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    lower = np.arange(n_bins, dtype=np.float64) / n_bins
    upper = np.arange(1, n_bins + 1, dtype=np.float64) / n_bins
    count = np.bincount(bin_index, minlength=n_bins)
    mean_confidence = np.zeros(n_bins, dtype=np.float64)
    accuracy = np.zeros(n_bins, dtype=np.float64)
    for index in range(n_bins):
        mask = bin_index == index
        if mask.any():
            mean_confidence[index] = confidence[mask].mean()
            accuracy[index] = correct[mask].mean()
    return ReliabilityData(lower, upper, count, mean_confidence, accuracy)


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    bins = reliability_data(probabilities, labels, n_bins=n_bins)
    total = bins.count.sum()
    if total == 0:
        raise ValueError("ECE is undefined for an empty dataset")
    weights = bins.count / total
    return float(np.sum(weights * np.abs(bins.accuracy - bins.mean_confidence)))


def predictive_entropy(probabilities: np.ndarray, normalized: bool = False) -> np.ndarray:
    validate_probabilities(probabilities)
    values = -np.sum(probabilities * np.log(np.clip(probabilities, EPS, 1.0)), axis=1)
    if normalized:
        values = values / math.log(probabilities.shape[1])
    return values


def classification_metrics(probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict[str, float]:
    validate_probabilities(probabilities)
    labels = np.asarray(labels, dtype=np.int64)
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float((predictions == labels).mean()),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "nll": negative_log_likelihood(probabilities, labels),
        "ece_15": expected_calibration_error(probabilities, labels, n_bins=n_bins),
        "brier": multiclass_brier_score(probabilities, labels),
    }


def binary_detection_metrics(positive: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """Return AUROC/AUPR when higher score means more positive-like."""
    positive = np.asarray(positive, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    if positive.size != score.size or positive.size == 0:
        raise ValueError("positive labels and scores must be non-empty and equally sized")
    if np.unique(positive).size != 2:
        raise ValueError("AUROC/AUPR require both positive and negative examples")
    if not np.isfinite(score).all():
        raise ValueError("detection scores contain NaN or infinity")
    return {
        "auroc": float(roc_auc_score(positive, score)),
        "aupr": float(average_precision_score(positive, score)),
    }


def error_detection_metrics(correct: np.ndarray, uncertainty: np.ndarray) -> dict[str, float]:
    """Prediction error is positive; higher uncertainty must mean more error-like."""
    correct = np.asarray(correct, dtype=bool)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    result = binary_detection_metrics((~correct).astype(np.int64), uncertainty)
    for name, mask in (("correct", correct), ("incorrect", ~correct)):
        values = uncertainty[mask]
        result[f"{name}_count"] = int(values.size)
        result[f"{name}_mean_uncertainty"] = float(values.mean()) if values.size else float("nan")
        result[f"{name}_median_uncertainty"] = float(np.median(values)) if values.size else float("nan")
    return result

