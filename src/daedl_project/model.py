from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


def _conv3x3(in_channels: int, out_channels: int, stride: int, use_spectral_norm: bool) -> nn.Conv2d:
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
    return spectral_norm(layer) if use_spectral_norm else layer


def _linear(in_features: int, out_features: int, use_spectral_norm: bool) -> nn.Linear:
    layer = nn.Linear(in_features, out_features)
    return spectral_norm(layer) if use_spectral_norm else layer


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int, use_spectral_norm: bool):
        super().__init__()
        self.conv1 = _conv3x3(in_channels, out_channels, stride, use_spectral_norm)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_channels, out_channels, 1, use_spectral_norm)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            if use_spectral_norm:
                projection = spectral_norm(projection)
            self.shortcut = nn.Sequential(projection, nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class CifarResNet18(nn.Module):
    """Compact CIFAR stem, ResNet-18 block pattern, and explicit 128-D embedding."""

    def __init__(
        self,
        num_classes: int = 10,
        embedding_dim: int = 128,
        base_width: int = 32,
        spectral_normalization: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.in_channels = base_width
        self.stem = nn.Sequential(
            _conv3x3(3, base_width, 1, spectral_normalization),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._stage(base_width, 2, 1, spectral_normalization)
        self.stage2 = self._stage(base_width * 2, 2, 2, spectral_normalization)
        self.stage3 = self._stage(base_width * 4, 2, 2, spectral_normalization)
        self.stage4 = self._stage(base_width * 8, 2, 2, spectral_normalization)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = _linear(base_width * 8, embedding_dim, spectral_normalization)
        self.embedding_activation = nn.ReLU(inplace=True)
        self.classifier = _linear(embedding_dim, num_classes, spectral_normalization)

    def _stage(self, out_channels: int, blocks: int, stride: int, sn: bool) -> nn.Sequential:
        layers = [BasicBlock(self.in_channels, out_channels, stride, sn)]
        self.in_channels = out_channels
        layers.extend(BasicBlock(out_channels, out_channels, 1, sn) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).flatten(1)
        embedding = self.embedding_activation(self.embedding(x))
        logits = self.classifier(embedding)
        return logits, embedding

