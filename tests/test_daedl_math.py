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


def test_nonfinite_logits_are_rejected():
    with pytest.raises(FloatingPointError):
        concentration_from_logits(torch.tensor([[float("inf"), 0.0]]))
