from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .reproducibility import DEFAULT_SEED, seed_worker, stratified_indices, stratified_subset


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return image, label, index


class IndexedSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: np.ndarray):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return self.indices.size

    def __getitem__(self, position: int):
        index = int(self.indices[position])
        image, label = self.dataset[index]
        return image, label, index


@dataclass
class CIFARLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
    density: DataLoader
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray


def train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def evaluation_transform() -> transforms.Compose:
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    )


def _loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int, num_workers: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        worker_init_fn=seed_worker if num_workers else None,
        generator=generator,
    )


def build_cifar10_loaders(
    data_dir: str | Path,
    batch_size: int = 64,
    split_seed: int = DEFAULT_SEED,
    loader_seed: int = DEFAULT_SEED,
    pilot_train_size: int | None = None,
    pilot_validation_size: int | None = None,
    pilot_test_size: int | None = None,
    num_workers: int = 0,
) -> CIFARLoaders:
    data_dir = Path(data_dir)
    augmented = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform())
    clean_train = datasets.CIFAR10(data_dir, train=True, download=False, transform=evaluation_transform())
    test_base = datasets.CIFAR10(data_dir, train=False, download=True, transform=evaluation_transform())

    labels = np.asarray(augmented.targets)
    train_idx, val_idx = stratified_indices(labels, validation_size=5_000, seed=split_seed)
    if pilot_train_size is not None:
        train_idx = stratified_subset(train_idx, labels, pilot_train_size, split_seed + 1)
    if pilot_validation_size is not None:
        val_idx = stratified_subset(val_idx, labels, pilot_validation_size, split_seed + 2)

    test_idx = np.arange(len(test_base))
    if pilot_test_size is not None:
        test_idx = stratified_subset(test_idx, test_base.targets, pilot_test_size, split_seed + 3)

    assert np.intersect1d(train_idx, val_idx).size == 0
    assert set(labels[train_idx]) == set(range(10))
    assert set(labels[val_idx]) == set(range(10))
    assert set(np.asarray(test_base.targets)[test_idx]) == set(range(10))

    train_set = IndexedSubset(augmented, train_idx)
    density_set = IndexedSubset(clean_train, train_idx)
    validation_set = IndexedSubset(clean_train, val_idx)
    test_set = IndexedSubset(test_base, test_idx)

    return CIFARLoaders(
        train=_loader(train_set, batch_size, True, loader_seed, num_workers),
        validation=_loader(validation_set, batch_size, False, loader_seed, num_workers),
        test=_loader(test_set, batch_size, False, loader_seed, num_workers),
        density=_loader(density_set, batch_size, False, loader_seed, num_workers),
        train_indices=train_idx,
        validation_indices=val_idx,
        test_indices=test_idx,
    )


def build_svhn_loader(
    data_dir: str | Path,
    batch_size: int,
    seed: int,
    subset_size: int | None = None,
    num_workers: int = 0,
) -> DataLoader:
    base = datasets.SVHN(data_dir, split="test", download=True, transform=evaluation_transform())
    if subset_size is None or subset_size >= len(base):
        dataset: Dataset = IndexedDataset(base)
    else:
        generator = np.random.default_rng(seed)
        indices = np.sort(generator.choice(len(base), size=subset_size, replace=False))
        dataset = IndexedSubset(base, indices)
    return _loader(dataset, batch_size, False, seed, num_workers)

