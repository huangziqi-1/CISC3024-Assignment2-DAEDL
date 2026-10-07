import numpy as np
import pytest
import torch

from daedl_project.losses import concentration_from_logits, daedl_loss
from daedl_project.metrics import validate_probabilities


def test_daedl_concentration_and_predictive_probability():
    logits = torch.tensor([[0.0, 1.0, -1.0], [2.0, 2.0, 2.0]])
    alpha = concentration_from_logits(logits)
    probabilities = alpha / alpha.sum(dim=1, keepdim=True)
    assert torch.all(alpha > 0)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))
    validate_probabilities(probabilities.numpy())
    assert torch.allclose(probabilities, torch.softmax(logits, dim=1))


def test_density_zero_produces_uniform_dirichlet():
    logits = torch.tensor([[4.0, -2.0, 1.0]])
    density = torch.tensor([[0.0]])
    alpha = torch.exp(logits * density)
    probabilities = alpha / alpha.sum(dim=1, keepdim=True)
    assert torch.allclose(alpha, torch.ones_like(alpha))
    assert torch.allclose(probabilities, torch.full_like(probabilities, 1 / 3))


def test_daedl_loss_and_gradients_are_finite():
    logits = torch.tensor([[0.2, -0.1, 0.4], [0.0, 0.3, -0.2]], requires_grad=True)
    labels = torch.tensor([2, 1])
    loss, components = daedl_loss(logits, labels, num_classes=3)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert all(np.isfinite(value) for value in components.values())


def test_mean_reduction_scales_complete_daedl_objective_consistently():
    logits = torch.tensor([[0.2, -0.4, 0.7], [-0.1, 0.6, 0.3]], dtype=torch.float64)
    labels = torch.tensor([2, 1])
    regularization = 0.05
    mean_loss, _ = daedl_loss(logits, labels, num_classes=3, regularization=regularization)
    alpha = torch.exp(logits)
    strength = alpha.sum(dim=1, keepdim=True)
    targets = torch.nn.functional.one_hot(labels, 3).to(torch.float64)
    error = ((targets - alpha / strength) ** 2).sum()
    variance = (alpha * (strength - alpha) / (strength.square() * (strength + 1.0))).sum()
    adjusted = alpha * (1.0 - targets) + targets
    alpha0 = adjusted.sum(dim=1)
    kl = (
        torch.lgamma(alpha0)
        - torch.lgamma(adjusted).sum(dim=1)
        - torch.lgamma(torch.tensor(3.0, dtype=torch.float64))
        + ((adjusted - 1.0) * (torch.digamma(adjusted) - torch.digamma(alpha0).unsqueeze(1))).sum(dim=1)
    ).sum()
    official_sum_form = error + variance + regularization * kl
    assert torch.allclose(mean_loss * labels.numel(), official_sum_form)


def test_nonfinite_logits_are_rejected():
    with pytest.raises(FloatingPointError):
        concentration_from_logits(torch.tensor([[float("inf"), 0.0]]))
