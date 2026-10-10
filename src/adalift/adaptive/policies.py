from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch


@dataclass(frozen=True, slots=True)
class ResidualHaltingPolicy:
    threshold: float = 0.01
    min_loops: int = 2
    max_loops: int = 16
    eps: float = 1e-8

    def validate(self) -> None:
        if not isfinite(self.threshold) or self.threshold < 0:
            raise ValueError("threshold must be finite and non-negative")
        if not isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be finite and positive")
        if not 1 <= self.min_loops <= self.max_loops:
            raise ValueError("require 1 <= min_loops <= max_loops")

    def residual(self, current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        self.validate()
        if current.shape != previous.shape:
            raise ValueError("current and previous shapes must match")
        delta = (current.float() - previous.float()).flatten(1).norm(dim=1)
        base = current.float().flatten(1).norm(dim=1).clamp_min(self.eps)
        result: torch.Tensor = delta / base
        return result

    def should_halt(
        self,
        current: torch.Tensor,
        previous: torch.Tensor,
        loop_index: int,
    ) -> torch.Tensor:
        self.validate()
        if loop_index < self.min_loops:
            return torch.zeros(current.shape[0], dtype=torch.bool, device=current.device)
        result: torch.Tensor = self.residual(current, previous) <= self.threshold
        return result
