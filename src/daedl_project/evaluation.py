from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .gda import GDAState, density_score, fit_gda, log_density
from .losses import LOGIT_CLAMP
from .metrics import (
    binary_detection_metrics,
    classification_metrics,
    error_detection_metrics,
    predictive_entropy,
)


@torch.no_grad()
def extract_embeddings(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    embeddings, labels_all, sample_indices = [], [], []
    for images, labels, indices in loader:
        logits, features = model(images.to(device))
        if not torch.isfinite(logits).all() or not torch.isfinite(features).all():
            raise FloatingPointError("non-finite model output during feature extraction")
        embeddings.append(features.detach().to(device="cpu", dtype=torch.float64))
        labels_all.append(labels.to(dtype=torch.int64))
        sample_indices.append(torch.as_tensor(indices, dtype=torch.int64))
    return torch.cat(embeddings), torch.cat(labels_all), torch.cat(sample_indices)


def fit_daedl_density(model: nn.Module, loader: DataLoader, device: torch.device) -> GDAState:
    embeddings, labels, _ = extract_embeddings(model, loader, device)
    if embeddings.device.type != "cpu" or embeddings.dtype != torch.float64:
        raise AssertionError("DAEDL density embeddings must be CPU float64")
    return fit_gda(embeddings, labels, num_classes=10)


def embedding_density_diagnostics(
    embeddings: torch.Tensor, labels: torch.Tensor, near_constant_relative: float = 1e-6
) -> dict:
    """Describe feature degeneracy and per-class empirical covariance stability."""
    x = embeddings.detach().to(device="cpu", dtype=torch.float64)
    y = labels.detach().to(device="cpu", dtype=torch.int64)
    standard_deviation = x.std(dim=0, unbiased=True)
    scale = max(float(standard_deviation.max()), 1e-12)
    constant = standard_deviation <= 1e-12
    near_constant = standard_deviation <= scale * near_constant_relative
    classes = []
    for class_index in range(10):
        class_x = x[y == class_index]
        centered = class_x - class_x.mean(dim=0)
        covariance = centered.T @ centered / (class_x.shape[0] - 1)
        eigenvalues = torch.linalg.eigvalsh(0.5 * (covariance + covariance.T))
        positive = eigenvalues[eigenvalues > max(float(eigenvalues.max()) * 1e-12, 1e-15)]
        condition = float(eigenvalues.max() / positive.min()) if positive.numel() else float("inf")
        classes.append(
            {
                "class": class_index,
                "samples": int(class_x.shape[0]),
                "centered_embedding_rank": int(torch.linalg.matrix_rank(centered)),
                "covariance_rank": int(torch.linalg.matrix_rank(covariance)),
                "minimum_eigenvalue": float(eigenvalues.min()),
                "maximum_eigenvalue": float(eigenvalues.max()),
                "positive_spectrum_condition_number": condition,
            }
        )
    return {
        "samples": int(x.shape[0]),
        "embedding_dim": int(x.shape[1]),
        "nonfinite_fraction": float((~torch.isfinite(x)).sum() / x.numel()),
        "constant_dimensions_threshold": 1e-12,
        "constant_dimensions": int(constant.sum()),
        "near_constant_relative_threshold": near_constant_relative,
        "near_constant_absolute_threshold": scale * near_constant_relative,
        "near_constant_dimensions": int(near_constant.sum()),
        "global_embedding_rank": int(torch.linalg.matrix_rank(x - x.mean(dim=0))),
        "classes": classes,
    }


def _row_base(index: int, label: int, probabilities: np.ndarray, logits: np.ndarray) -> dict:
    prediction = int(probabilities.argmax())
    row = {
        "sample_index": index,
        "ground_truth": label,
        "predicted_label": prediction,
        "max_confidence": float(probabilities.max()),
        "correct": bool(prediction == label),
    }
    for class_index, value in enumerate(logits):
        row[f"logit_{class_index}"] = float(value)
    for class_index, value in enumerate(probabilities):
        row[f"probability_{class_index}"] = float(value)
    return row


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_type: str,
    gda_state: GDAState | None = None,
) -> tuple[dict, list[dict]]:
    model.eval()
    rows: list[dict] = []
    all_probabilities, all_labels = [], []
    for images, labels, indices in loader:
        logits, embeddings = model(images.to(device))
        if model_type == "softmax":
            probabilities_t = F.softmax(logits, dim=1)
            alpha_t = None
            density_log_t = density_t = None
        elif model_type == "daedl":
            if gda_state is None:
                raise ValueError("DAEDL evaluation requires a fitted GDA state")
            density_log_t = log_density(embeddings, gda_state)
            density_t = density_score(density_log_t, gda_state)
            density_device = density_t.to(device=device, dtype=logits.dtype)
            scaled_logits = torch.clamp(
                logits * density_device.unsqueeze(1), min=-LOGIT_CLAMP, max=LOGIT_CLAMP
            )
            alpha_t = torch.exp(scaled_logits)
            probabilities_t = alpha_t / alpha_t.sum(dim=1, keepdim=True)
        else:
            raise ValueError(f"unsupported model type: {model_type}")

        probabilities = probabilities_t.detach().cpu().numpy()
        logits_np = logits.detach().cpu().numpy()
        labels_np = labels.numpy()
        indices_np = np.asarray(indices)
        entropy = predictive_entropy(probabilities, normalized=False)
        normalized_entropy = entropy / math.log(probabilities.shape[1])
        for position in range(labels_np.size):
            row = _row_base(
                int(indices_np[position]), int(labels_np[position]), probabilities[position], logits_np[position]
            )
            row["predictive_entropy"] = float(entropy[position])
            row["normalized_predictive_entropy"] = float(normalized_entropy[position])
            if model_type == "daedl":
                alpha = alpha_t[position].detach().cpu().numpy()
                row["log_feature_density"] = float(density_log_t[position])
                row["density_score"] = float(density_t[position])
                row["dirichlet_strength"] = float(alpha.sum())
                for class_index, value in enumerate(alpha):
                    row[f"alpha_{class_index}"] = float(value)
            rows.append(row)
        all_probabilities.append(probabilities)
        all_labels.append(labels_np)

    probabilities = np.concatenate(all_probabilities)
    labels_array = np.concatenate(all_labels)
    metrics = classification_metrics(probabilities, labels_array)
    correct = np.asarray([row["correct"] for row in rows])
    uncertainty = np.asarray([row["normalized_predictive_entropy"] for row in rows])
    if np.unique(correct).size == 2:
        metrics["error_detection"] = error_detection_metrics(correct, uncertainty)
    else:
        metrics["error_detection"] = {"status": "undefined: pilot predictions contain one correctness class"}
    return metrics, rows


def ood_metrics_from_rows(id_rows: list[dict], ood_rows: list[dict], score_key: str) -> dict[str, float]:
    """OOD is positive and score must increase for more OOD-like samples."""
    labels = np.concatenate([np.zeros(len(id_rows), dtype=int), np.ones(len(ood_rows), dtype=int)])
    scores = np.asarray([row[score_key] for row in id_rows] + [row[score_key] for row in ood_rows])
    return binary_detection_metrics(labels, scores)


@torch.no_grad()
def score_ood_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_type: str,
    gda_state: GDAState | None = None,
) -> list[dict]:
    """Save OOD-compatible scores without treating the OOD dataset's labels as CIFAR classes."""
    model.eval()
    rows: list[dict] = []
    for images, source_labels, indices in loader:
        logits, embeddings = model(images.to(device))
        if model_type == "softmax":
            probabilities_t = F.softmax(logits, dim=1)
            density_log_t = density_t = alpha_t = None
        elif model_type == "daedl":
            if gda_state is None:
                raise ValueError("DAEDL OOD scoring requires a fitted GDA state")
            density_log_t = log_density(embeddings, gda_state)
            density_t = density_score(density_log_t, gda_state)
            scaled = torch.clamp(
                logits * density_t.to(device=device, dtype=logits.dtype).unsqueeze(1),
                min=-LOGIT_CLAMP,
                max=LOGIT_CLAMP,
            )
            alpha_t = torch.exp(scaled)
            probabilities_t = alpha_t / alpha_t.sum(dim=1, keepdim=True)
        else:
            raise ValueError(f"unsupported model type: {model_type}")

        probabilities = probabilities_t.detach().cpu().numpy()
        entropy = predictive_entropy(probabilities, normalized=False)
        normalized_entropy = entropy / math.log(probabilities.shape[1])
        for position, sample_index in enumerate(np.asarray(indices)):
            row = {
                "sample_index": int(sample_index),
                "source_dataset_label_unused": int(source_labels[position]),
                "predicted_label": int(probabilities[position].argmax()),
                "max_confidence": float(probabilities[position].max()),
                "predictive_entropy": float(entropy[position]),
                "normalized_predictive_entropy": float(normalized_entropy[position]),
            }
            for class_index, value in enumerate(probabilities[position]):
                row[f"probability_{class_index}"] = float(value)
            if model_type == "daedl":
                row["log_feature_density"] = float(density_log_t[position])
                row["density_score"] = float(density_t[position])
                row["dirichlet_strength"] = float(alpha_t[position].sum().detach().cpu())
            rows.append(row)
    return rows
