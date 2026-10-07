import math

import numpy as np
import pytest
from sklearn.metrics import log_loss

from src.daedl_project.metrics import (
    binary_detection_metrics,
    expected_calibration_error,
    multiclass_brier_score,
    negative_log_likelihood,
    predictive_entropy,
    reliability_data,
)


def test_calibration_and_proper_scores_known_case():
    probabilities = np.array([[1.0, 0.0], [0.5, 0.5]])
    labels = np.array([0, 1])
    assert expected_calibration_error(probabilities, labels, n_bins=2) == pytest.approx(0.25)
    assert multiclass_brier_score(probabilities, labels) == pytest.approx(0.25)
    assert negative_log_likelihood(probabilities, labels) == pytest.approx(log_loss(labels, probabilities))


def test_reliability_handles_empty_bins_and_confidence_one():
    probabilities = np.array([[1.0, 0.0], [0.9, 0.1]])
    bins = reliability_data(probabilities, np.array([0, 0]), n_bins=15)
    assert bins.count.sum() == 2
    assert bins.count[-1] == 1  # confidence exactly 1.0 stays in the closed final bin
    assert bins.count[13] == 1  # confidence 0.9 is in [13/15, 14/15)
    assert np.isfinite(bins.mean_confidence).all()


def test_predictive_entropy_definitions():
    probabilities = np.array([[0.5, 0.5], [1.0, 0.0]])
    raw = predictive_entropy(probabilities)
    normalized = predictive_entropy(probabilities, normalized=True)
    assert raw[0] == pytest.approx(math.log(2))
    assert normalized[0] == pytest.approx(1.0)
    assert normalized[1] == pytest.approx(0.0)


def test_ood_positive_score_direction():
    # OOD is positive and receives larger scores, so both metrics must be perfect.
    positive = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9])
    result = binary_detection_metrics(positive, score)
    assert result["auroc"] == pytest.approx(1.0)
    assert result["aupr"] == pytest.approx(1.0)


def test_detection_rejects_single_class():
    with pytest.raises(ValueError, match="both positive and negative"):
        binary_detection_metrics(np.ones(3), np.arange(3))
