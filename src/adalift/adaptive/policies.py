from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ResidualHaltingPolicy:
    threshold: float = 0.01
    min_loops: int = 2
    max_loops: int = 16
    eps: float = 1e-8

    def validate(self) -> None:
        if self.threshold < 0:
            raise ValueError("threshold must be non-negative")
        if not 1 <= self.min_loops <= self.max_loops:
            raise ValueError("require 1 <= min_loops <= max_loops")

    def residual(self, current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        self.validate()
        if current.shape != previous.shape:
            raise ValueError("current and previous shapes must match")
        delta = (current - previous).flatten(1).norm(dim=1)
        base = current.flatten(1).norm(dim=1).clamp_min(self.eps)
        return delta / base

    def should_halt(
        self,
        current: torch.Tensor,
        previous: torch.Tensor,
        loop_index: int,
    ) -> torch.Tensor:
        if loop_index < self.min_loops:
            return torch.zeros(current.shape[0], dtype=torch.bool, device=current.device)
        return self.residual(current, previous) <= self.threshold
