from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from adalift.training.ema import EMA


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    ema: EMA | None = None,
    step: int = 0,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Atomically save training state using ``torch.save``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    state: dict[str, Any] = {
        "model": model.state_dict(),
        "step": step,
        "metadata": metadata or {},
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if ema is not None:
        state["ema"] = ema.state_dict()
    torch.save(state, tmp)
    os.replace(tmp, target)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    ema: EMA | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if ema is not None and "ema" in state:
        ema.load_state_dict(state["ema"])
    return {"step": int(state.get("step", 0)), "metadata": state.get("metadata", {})}
