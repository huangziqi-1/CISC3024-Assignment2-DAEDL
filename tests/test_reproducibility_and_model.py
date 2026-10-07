import numpy as np
import torch

from src.daedl_project.model import CifarResNet18
from src.daedl_project.reproducibility import stratified_indices, stratified_subset


def test_split_is_reproducible_stratified_and_disjoint():
    labels = np.repeat(np.arange(10), 100)
    train_a, val_a = stratified_indices(labels, validation_size=200, seed=3024)
    train_b, val_b = stratified_indices(labels, validation_size=200, seed=3024)
    assert np.array_equal(train_a, train_b)
    assert np.array_equal(val_a, val_b)
    assert np.intersect1d(train_a, val_a).size == 0
    assert np.bincount(labels[val_a], minlength=10).tolist() == [20] * 10
    subset = stratified_subset(train_a, labels, size=100, seed=3025)
    assert np.bincount(labels[subset], minlength=10).tolist() == [10] * 10


def test_cifar_resnet_exposes_logits_and_128d_embedding():
    model = CifarResNet18(embedding_dim=128, base_width=8)
    logits, embedding = model(torch.randn(2, 3, 32, 32))
    assert logits.shape == (2, 10)
    assert embedding.shape == (2, 128)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(embedding).all()

