from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .losses import daedl_loss
from .metrics import classification_metrics


@dataclass
class TrainingResult:
    history: list[dict]
    best_state: dict[str, torch.Tensor]
    final_state: dict[str, torch.Tensor]
    best_epoch: int
    best_validation_nll: float
    runtime_seconds: float


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _batch_loss(model_type: str, logits: torch.Tensor, labels: torch.Tensor, regularization: float):
    if model_type == "softmax":
        loss = F.cross_entropy(logits, labels)
        return loss, {}
    if model_type == "daedl":
        return daedl_loss(logits, labels, regularization=regularization)
    raise ValueError(f"unsupported model type: {model_type}")


@torch.no_grad()
def evaluate_training_stage(
    model: nn.Module, loader: DataLoader, device: torch.device, model_type: str, regularization: float
) -> dict[str, float]:
    model.eval()
    losses, probabilities, labels_all = [], [], []
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        logits, _ = model(images)
        loss, _ = _batch_loss(model_type, logits, labels, regularization)
        if not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise FloatingPointError("non-finite validation output")
        losses.append(float(loss.detach().cpu()) * labels.size(0))
        probabilities.append(F.softmax(logits, dim=1).detach().cpu().numpy())
        labels_all.append(labels.detach().cpu().numpy())
    probs = np.concatenate(probabilities)
    targets = np.concatenate(labels_all)
    result = classification_metrics(probs, targets)
    result["loss"] = float(sum(losses) / targets.size)
    return result


def train_model(
    model: nn.Module,
    model_type: str,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    regularization: float,
    minimum_learning_rate: float = 0.0,
    progress_log_path: str | Path | None = None,
) -> TrainingResult:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=minimum_learning_rate
    )
    best_nll = float("inf")
    best_epoch = 0
    best_state = None
    history: list[dict] = []
    model.to(device)
    _synchronize(device)
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        loss_total, correct, count = 0.0, 0, 0
        component_totals: dict[str, float] = {}
        epoch_start = time.perf_counter()
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(images)
            loss, components = _batch_loss(model_type, logits, labels, regularization)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training value at epoch {epoch}")
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"non-finite gradient at epoch {epoch}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            batch_size = labels.size(0)
            loss_total += float(loss.detach().cpu()) * batch_size
            correct += int((logits.argmax(dim=1) == labels).sum())
            count += batch_size
            for key, value in components.items():
                component_totals[key] = component_totals.get(key, 0.0) + value * batch_size

        scheduler.step()
        validation = evaluate_training_stage(model, validation_loader, device, model_type, regularization)
        _synchronize(device)
        row = {
            "epoch": epoch,
            "train_loss": loss_total / count,
            "train_accuracy": correct / count,
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "validation_macro_f1": validation["macro_f1"],
            "validation_nll": validation["nll"],
            "validation_ece_15": validation["ece_15"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        row.update({f"train_{key}": value / count for key, value in component_totals.items()})
        history.append(row)
        if progress_log_path is not None:
            import json

            with Path(progress_log_path).open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        if validation["nll"] < best_nll:
            best_nll = validation["nll"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    _synchronize(device)
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    final_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    return TrainingResult(
        history, best_state, final_state, best_epoch, best_nll, time.perf_counter() - started
    )
