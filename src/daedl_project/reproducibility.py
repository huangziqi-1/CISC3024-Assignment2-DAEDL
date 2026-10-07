from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import train_test_split


DEFAULT_SEED = 3024


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    """Seed Python, NumPy, and PyTorch without claiming perfect MPS determinism."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_indices(
    labels: list[int] | np.ndarray,
    validation_size: int,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    labels_array = np.asarray(labels)
    indices = np.arange(labels_array.size)
    train_idx, validation_idx = train_test_split(
        indices,
        test_size=validation_size,
        random_state=seed,
        shuffle=True,
        stratify=labels_array,
    )
    return np.sort(train_idx), np.sort(validation_idx)


def stratified_subset(
    indices: np.ndarray,
    labels: list[int] | np.ndarray,
    size: int,
    seed: int,
) -> np.ndarray:
    indices = np.asarray(indices)
    labels_array = np.asarray(labels)[indices]
    if size >= indices.size:
        return np.sort(indices)
    selected, _ = train_test_split(
        indices,
        train_size=size,
        random_state=seed,
        shuffle=True,
        stratify=labels_array,
    )
    return np.sort(selected)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@dataclass(frozen=True)
class DeviceChoice:
    requested: str
    effective: str
    mps_built: bool
    mps_available: bool
    cuda_available: bool


def choose_device(requested: str, allow_cpu_fallback: bool = False) -> DeviceChoice:
    requested = requested.lower()
    mps_built = torch.backends.mps.is_built()
    mps_available = torch.backends.mps.is_available()
    cuda_available = torch.cuda.is_available()

    if requested == "mps":
        if not mps_available:
            if not allow_cpu_fallback:
                raise RuntimeError("MPS was requested but is unavailable; CPU fallback was not authorized.")
            effective = "cpu"
        else:
            effective = "mps"
    elif requested == "cuda":
        if not cuda_available:
            if not allow_cpu_fallback:
                raise RuntimeError("CUDA was requested but is unavailable; CPU fallback was not authorized.")
            effective = "cpu"
        else:
            effective = "cuda"
    elif requested == "cpu":
        effective = "cpu"
    else:
        raise ValueError("device must be one of: mps, cuda, cpu")

    return DeviceChoice(requested, effective, mps_built, mps_available, cuda_available)

