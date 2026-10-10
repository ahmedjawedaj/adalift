from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ModelConfig:
    in_channels: int = 4
    input_size: int = 32
    patch_size: int = 2
    hidden_size: int = 768
    depth: int = 12
    prelude_depth: int = 4
    recurrent_depth: int = 2
    num_heads: int = 12
    mlp_ratio: float = 4.0
    num_classes: int = 1000
    class_dropout_prob: float = 0.0
    qk_norm: bool = True
    rope_theta: float = 10_000.0
    depth_embed_freq_dim: int = 256
    time_embed_freq_dim: int = 256

    def validate(self) -> None:
        if self.input_size % self.patch_size != 0:
            raise ValueError("input_size must be divisible by patch_size")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        head_dim = self.hidden_size // self.num_heads
        if head_dim % 4 != 0:
            raise ValueError("attention head dimension must be divisible by 4 for 2D RoPE")
        if self.prelude_depth < 1:
            raise ValueError("prelude_depth must be >= 1")
        if self.recurrent_depth < 1:
            raise ValueError("recurrent_depth must be >= 1")
        if self.prelude_depth > self.depth:
            raise ValueError("prelude_depth cannot exceed nominal depth")
        if self.num_classes < 1:
            raise ValueError("num_classes must be >= 1")


@dataclass(slots=True)
class TrainConfig:
    k_train: int = 2
    prelude_loss_weight: float = 0.01
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    grad_clip_norm: float | None = 1.0
    ema_decay: float = 0.9999
    seed: int = 42

    def validate(self) -> None:
        if self.k_train < 1:
            raise ValueError("k_train must be >= 1")
        if self.prelude_loss_weight < 0:
            raise ValueError("prelude_loss_weight must be non-negative")
        if not 0.0 < self.learning_rate:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")


@dataclass(slots=True)
class SamplingConfig:
    num_steps: int = 50
    num_loops: int = 4

    def validate(self) -> None:
        if self.num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if self.num_loops < 1:
            raise ValueError("num_loops must be >= 1")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def save_yaml(data: object, path: str | Path) -> None:
    payload = asdict(data) if hasattr(data, "__dataclass_fields__") else data
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
