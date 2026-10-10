from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_embedding(values: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """Embed scalar values with sinusoidal features.

    Args:
        values: Shape ``[batch]``.
        dim: Output feature dimension.
        max_period: Controls the minimum frequency.
    """
    if values.ndim != 1:
        raise ValueError(f"values must have shape [batch], got {tuple(values.shape)}")
    half = dim // 2
    if half == 0:
        raise ValueError("embedding dim must be >= 2")
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=values.device, dtype=torch.float32)
        / max(half, 1)
    )
    args = values.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb.to(dtype=values.dtype if values.is_floating_point() else torch.float32)


class ScalarConditionEmbedding(nn.Module):
    def __init__(self, hidden_size: int, frequency_dim: int = 256) -> None:
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        emb = sinusoidal_embedding(value, self.frequency_dim)
        return self.mlp(emb.to(dtype=self.mlp[0].weight.dtype))


class LabelEmbedding(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float = 0.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        extra = 1 if dropout_prob > 0 else 0
        self.embedding = nn.Embedding(num_classes + extra, hidden_size)
        self.null_label = num_classes if extra else None

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        if labels.ndim != 1:
            raise ValueError("labels must have shape [batch]")
        labels = labels.long()
        if self.training and self.dropout_prob > 0:
            mask = torch.rand(labels.shape, device=labels.device) < self.dropout_prob
            labels = torch.where(mask, torch.full_like(labels, self.null_label), labels)
        return self.embedding(labels)


class PatchEmbed2d(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if x.ndim != 4:
            raise ValueError("x must have shape [batch, channels, height, width]")
        h = self.proj(x)
        grid = (h.shape[-2], h.shape[-1])
        tokens = h.flatten(2).transpose(1, 2)
        return tokens, grid
