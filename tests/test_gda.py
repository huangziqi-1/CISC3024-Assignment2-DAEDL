import pytest
import torch

from src.daedl_project.gda import class_log_probabilities, density_score, fit_gda, log_density


def test_gda_shapes_cpu_float64_and_density_range():
    generator = torch.Generator().manual_seed(3024)
    class_zero = torch.randn(30, 4, generator=generator) * 0.2 - 1.0
    class_one = torch.randn(30, 4, generator=generator) * 0.2 + 1.0
    embeddings = torch.cat([class_zero, class_one])
    labels = torch.cat([torch.zeros(30, dtype=torch.long), torch.ones(30, dtype=torch.long)])
    state = fit_gda(embeddings, labels, num_classes=2)
    assert state.means.shape == (2, 4)
    assert state.cholesky.shape == (2, 4, 4)
    assert state.means.device.type == "cpu"
    assert state.means.dtype == torch.float64
    assert class_log_probabilities(embeddings[:5], state).shape == (5, 2)
    values = log_density(embeddings, state)
    score = density_score(values, state)
    assert torch.isfinite(values).all()
    assert torch.all((score >= 0) & (score <= 1))


def test_gda_rejects_too_few_samples_for_full_covariance():
    with pytest.raises(ValueError, match="need more than embedding dimension"):
        fit_gda(torch.randn(6, 4), torch.tensor([0, 0, 0, 1, 1, 1]), num_classes=2)

