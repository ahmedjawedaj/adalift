from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
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

    def state_dict(self) -> dict[str, object]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.decay = float(state["decay"])
        shadow = state["shadow"]
        if not isinstance(shadow, dict):
            raise TypeError("EMA shadow state must be a dict")
        self.shadow = {str(k): v for k, v in shadow.items() if isinstance(v, torch.Tensor)}

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
