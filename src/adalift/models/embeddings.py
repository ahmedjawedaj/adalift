from __future__ import annotations

import math

import torch
from torch import nn


def validate_condition_inputs(
    x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> None:
    if x.ndim != 4 or x.shape[0] < 1 or not x.is_floating_point():
        raise ValueError(
            "x must be a nonempty floating-point [batch, channels, height, width] tensor"
        )
    if t.shape != (x.shape[0],) or labels.shape != (x.shape[0],):
        raise ValueError("t and labels must have shape [batch]; broadcasting is not supported")
    if t.device != x.device or labels.device != x.device:
        raise ValueError("x, t and labels must share a device")
    if not t.is_floating_point() or not bool(torch.isfinite(t).all()):
        raise ValueError("t must be finite floating-point values")
    if not bool(((t >= 0) & (t <= 1)).all()):
        raise ValueError("t must be in [0, 1]")
    if labels.dtype not in (torch.int32, torch.int64):
        raise ValueError("labels must be integer class indices")
    if not bool(((labels >= 0) & (labels < num_classes)).all()):
        raise ValueError("labels must be in [0, num_classes)")


def sinusoidal_embedding(
    values: torch.Tensor, dim: int, max_period: float = 10_000.0
) -> torch.Tensor:
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
        first_layer = self.mlp[0]
        assert isinstance(first_layer, nn.Linear)
        result: torch.Tensor = self.mlp(emb.to(dtype=first_layer.weight.dtype))
        return result


class LabelEmbedding(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float = 0.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        extra = 1 if dropout_prob > 0 else 0
        self.embedding = nn.Embedding(num_classes + extra, hidden_size)
        self.null_label = num_classes if extra else None

    def forward(
        self, labels: torch.Tensor, *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        if labels.ndim != 1:
            raise ValueError("labels must have shape [batch]")
        labels = labels.long()
        if self.training and self.dropout_prob > 0:
            assert self.null_label is not None
            mask = (
                torch.rand(labels.shape, device=labels.device, generator=generator)
                < self.dropout_prob
            )
            labels = torch.where(mask, torch.full_like(labels, self.null_label), labels)
        result: torch.Tensor = self.embedding(labels)
        return result


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
        if x.shape[-2] % self.patch_size or x.shape[-1] % self.patch_size:
            raise ValueError("spatial dimensions must be divisible by patch_size")
        h = self.proj(x)
        grid = (h.shape[-2], h.shape[-1])
        tokens = h.flatten(2).transpose(1, 2)
        return tokens, grid
