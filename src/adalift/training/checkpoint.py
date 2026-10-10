from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from adalift.training.ema import EMA


def _validate_metadata(value: Any) -> None:
    if value is None or isinstance(value, bool | int | float | str):
        return
    if isinstance(value, list | tuple):
        for item in value:
            _validate_metadata(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _validate_metadata(item)
        return
    raise TypeError("metadata must contain only plain values, lists, tuples and string-keyed dicts")


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    ema: EMA | None = None,
    step: int = 0,
    metadata: dict[str, Any] | None = None,
    capture_rng: bool = False,
    generator: torch.Generator | None = None,
) -> None:
    """Atomically save training state using ``torch.save``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _validate_metadata(metadata)
    state: dict[str, Any] = {
        "format_version": 1,
        "model": model.state_dict(),
        "step": step,
        "metadata": metadata or {},
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if ema is not None:
        state["ema"] = ema.state_dict()
    if capture_rng:
        state["rng"] = {
            "torch": torch.get_rng_state(),
            "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
            "generator": generator.get_state() if generator is not None else None,
        }
    elif generator is not None:
        raise ValueError("generator requires capture_rng=True")
    descriptor, tmp_name = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    try:
        torch.save(state, tmp_name)
        os.replace(tmp_name, target)
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    ema: EMA | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = False,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Load safe state dictionaries; never fall back to unrestricted pickle.

    Requested training states must be present. RNG capture/restore is opt-in;
    callers must also preserve data-loader/sampler progress and external RNGs.
    """
    state = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(state, dict) or "model" not in state:
        raise ValueError("checkpoint must contain a model state dict")
    if state.get("format_version", 0) not in (0, 1):
        raise ValueError("unsupported checkpoint format version")
    for name, requested in (
        ("optimizer", optimizer is not None),
        ("ema", ema is not None),
        ("rng", restore_rng),
    ):
        if requested and name not in state:
            raise ValueError(f"checkpoint is missing requested {name} state")
    if generator is not None and not restore_rng:
        raise ValueError("generator requires restore_rng=True")
    if restore_rng and generator is not None and state["rng"]["generator"] is None:
        raise ValueError("checkpoint is missing requested generator state")
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if ema is not None:
        ema.load_state_dict(state["ema"], model)
    if restore_rng:
        rng = state["rng"]
        torch.set_rng_state(rng["torch"].cpu())
        random.setstate(rng["python"])
        if rng["cuda"]:
            if not torch.cuda.is_available() or len(rng["cuda"]) != torch.cuda.device_count():
                raise ValueError("saved CUDA RNG states require matching available CUDA devices")
            torch.cuda.set_rng_state_all([value.cpu() for value in rng["cuda"]])
        if generator is not None:
            generator.set_state(rng["generator"].cpu())
    return {"step": int(state.get("step", 0)), "metadata": state.get("metadata", {})}
