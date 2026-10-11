from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from math import isfinite
from typing import Any

import torch
from torch import nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name].lerp_(param.detach(), 1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def _targets(self, model: nn.Module | None) -> dict[str, torch.Tensor]:
        if model is None:
            return self.shadow
        return {name: param for name, param in model.named_parameters() if param.requires_grad}

    def validate_state_dict(self, state: dict[str, Any], model: nn.Module | None = None) -> float:
        """Check a saved EMA state against ``model`` without mutating anything."""
        if not isinstance(state, dict) or "decay" not in state or "shadow" not in state:
            raise ValueError("EMA state must contain decay and shadow")
        decay = float(state["decay"])
        if not isfinite(decay) or not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0, 1)")
        shadow = state["shadow"]
        if not isinstance(shadow, dict):
            raise TypeError("EMA shadow state must be a dict")
        targets = self._targets(model)
        if set(shadow) != set(targets):
            raise ValueError("EMA parameter keys do not match the model")
        for name, target in targets.items():
            value = shadow[name]
            if not isinstance(value, torch.Tensor) or value.shape != target.shape:
                raise ValueError(f"EMA tensor shape mismatch: {name}")
        return decay

    def load_state_dict(self, state: dict[str, Any], model: nn.Module | None = None) -> None:
        decay = self.validate_state_dict(state, model)
        targets = self._targets(model)
        loaded = {
            name: state["shadow"][name]
            .detach()
            .to(device=target.device, dtype=target.dtype)
            .clone()
            for name, target in targets.items()
        }
        self.decay = decay
        self.shadow = loaded

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        backup: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    backup[name] = param.detach().clone()
                    param.copy_(self.shadow[name])
        try:
            yield
        finally:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in backup:
                        param.copy_(backup[name])
