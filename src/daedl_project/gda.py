from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GDAState:
    means: torch.Tensor
    cholesky: torch.Tensor
    log_weights: torch.Tensor
    log_density_min: float
    log_density_max: float
    jitter_by_class: list[float]

    def validate(self) -> None:
        tensors = (self.means, self.cholesky, self.log_weights)
        if any(t.device.type != "cpu" for t in tensors):
            raise AssertionError("GDA state must remain on CPU")
        if any(t.dtype != torch.float64 for t in tensors):
            raise AssertionError("GDA state must use float64")
        if not all(torch.isfinite(t).all() for t in tensors):
            raise FloatingPointError("GDA state contains NaN or infinity")
        if not self.log_density_max > self.log_density_min:
            raise ValueError("training density range is degenerate")


def _regularized_cholesky(covariance: torch.Tensor) -> tuple[torch.Tensor, float]:
    dimension = covariance.shape[0]
    scale = max(float(torch.trace(covariance) / dimension), 1e-12)
    eye = torch.eye(dimension, dtype=torch.float64)
    for multiplier in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        jitter = scale * multiplier
        factor, info = torch.linalg.cholesky_ex(covariance + jitter * eye)
        if int(info) == 0 and torch.isfinite(factor).all():
            return factor, jitter
    raise RuntimeError("GDA covariance remained non-positive-definite after adaptive jitter")


def class_log_probabilities(embeddings: torch.Tensor, state: GDAState) -> torch.Tensor:
    state.validate()
    x = embeddings.detach().to(device="cpu", dtype=torch.float64)
    if x.ndim != 2 or x.shape[1] != state.means.shape[1]:
        raise ValueError("embedding shape does not match fitted GDA")
    difference = x[:, None, :] - state.means[None, :, :]
    # solve_triangular expects [..., D, RHS]; arrange one solve per sample/class.
    rhs = difference.unsqueeze(-1)
    factors = state.cholesky.unsqueeze(0).expand(x.shape[0], -1, -1, -1)
    solved = torch.linalg.solve_triangular(factors, rhs, upper=False).squeeze(-1)
    mahalanobis = solved.square().sum(dim=-1)
    log_det = 2.0 * torch.log(torch.diagonal(state.cholesky, dim1=-2, dim2=-1)).sum(dim=1)
    normalizer = x.shape[1] * torch.log(torch.tensor(2.0 * torch.pi, dtype=torch.float64))
    return state.log_weights[None, :] - 0.5 * (normalizer + log_det[None, :] + mahalanobis)


def log_density(embeddings: torch.Tensor, state: GDAState) -> torch.Tensor:
    values = torch.logsumexp(class_log_probabilities(embeddings, state), dim=1)
    if not torch.isfinite(values).all():
        raise FloatingPointError("GDA produced non-finite log density")
    return values


def density_score(log_density_values: torch.Tensor, state: GDAState) -> torch.Tensor:
    values = log_density_values.detach().to(device="cpu", dtype=torch.float64)
    score = (values - state.log_density_min) / (state.log_density_max - state.log_density_min)
    score = torch.clamp(score, 0.0, 1.0)
    if not torch.isfinite(score).all():
        raise FloatingPointError("normalized density contains NaN or infinity")
    return score


def fit_gda(embeddings: torch.Tensor, labels: torch.Tensor, num_classes: int = 10) -> GDAState:
    x = embeddings.detach().to(device="cpu", dtype=torch.float64)
    y = labels.detach().to(device="cpu", dtype=torch.int64)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("invalid embedding/label shapes")
    means, factors, weights, jitters = [], [], [], []
    for class_index in range(num_classes):
        class_x = x[y == class_index]
        if class_x.shape[0] <= x.shape[1]:
            raise ValueError(
                f"class {class_index} has {class_x.shape[0]} samples; need more than embedding dimension {x.shape[1]}"
            )
        mean = class_x.mean(dim=0)
        centered = class_x - mean
        covariance = centered.T @ centered / (class_x.shape[0] - 1)
        factor, jitter = _regularized_cholesky(covariance)
        means.append(mean)
        factors.append(factor)
        weights.append(class_x.shape[0] / x.shape[0])
        jitters.append(jitter)

    provisional = GDAState(
        means=torch.stack(means),
        cholesky=torch.stack(factors),
        log_weights=torch.log(torch.tensor(weights, dtype=torch.float64)),
        log_density_min=0.0,
        log_density_max=1.0,
        jitter_by_class=jitters,
    )
    training_density = torch.logsumexp(class_log_probabilities(x, provisional), dim=1)
    provisional.log_density_min = float(training_density.min())
    provisional.log_density_max = float(training_density.max())
    provisional.validate()
    return provisional

