from __future__ import annotations

import torch
import torch.nn.functional as F


LOGIT_CLAMP = 30.0


def concentration_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Paper parameterization alpha=exp(logits), with a documented overflow guard."""
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite logits before DAEDL concentration construction")
    alpha = torch.exp(torch.clamp(logits, min=-LOGIT_CLAMP, max=LOGIT_CLAMP))
    if not torch.isfinite(alpha).all() or not torch.all(alpha > 0):
        raise FloatingPointError("Invalid DAEDL concentration parameters")
    return alpha


def dirichlet_kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    num_classes = alpha.shape[1]
    alpha0 = alpha.sum(dim=1)
    return (
        torch.lgamma(alpha0)
        - torch.lgamma(alpha).sum(dim=1)
        - torch.lgamma(torch.tensor(float(num_classes), device=alpha.device, dtype=alpha.dtype))
        + ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(alpha0).unsqueeze(1))).sum(dim=1)
    )


def daedl_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 10,
    regularization: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Expected MSE under Dir(alpha) plus the paper's misleading-evidence KL term."""
    alpha = concentration_from_logits(logits)
    strength = alpha.sum(dim=1, keepdim=True)
    targets = F.one_hot(labels, num_classes=num_classes).to(dtype=alpha.dtype)
    probabilities = alpha / strength
    error = ((targets - probabilities) ** 2).sum(dim=1)
    variance = (alpha * (strength - alpha) / (strength.square() * (strength + 1.0))).sum(dim=1)
    adjusted_alpha = alpha * (1.0 - targets) + targets
    kl = dirichlet_kl_to_uniform(adjusted_alpha)
    per_sample = error + variance + regularization * kl
    loss = per_sample.mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite DAEDL loss")
    return loss, {
        "expected_error": float(error.mean().detach().cpu()),
        "predictive_variance": float(variance.mean().detach().cpu()),
        "kl": float(kl.mean().detach().cpu()),
    }

