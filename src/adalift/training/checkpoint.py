from __future__ import annotations

import copy
import os
import random
import tempfile
from math import isfinite
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
    Validation precedes application; failures restore caller state. Custom load
    hooks must not have irreversible external side effects.
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
    # Validate everything that can be validated before mutating any caller object.
    result = {"step": int(state.get("step", 0)), "metadata": state.get("metadata", {})}
    _validate_metadata(result["metadata"])
    _validate_model_state(model, state["model"])
    if optimizer is not None:
        _validate_optimizer_state(optimizer, state["optimizer"])
    if ema is not None:
        ema.validate_state_dict(state["ema"], model)
    rng = _validate_rng_state(state["rng"], generator) if restore_rng else None

    # Apply with rollback so a failure part-way leaves the caller's objects unchanged.
    model_backup = copy.deepcopy(model.state_dict())
    optimizer_backup = copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None
    ema_backup = (ema.decay, copy.deepcopy(ema.shadow)) if ema is not None else None
    rng_backup = (
        {
            "torch": torch.get_rng_state(),
            "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
            "generator": generator.get_state() if generator is not None else None,
        }
        if restore_rng
        else None
    )
    try:
        model.load_state_dict(state["model"])
        if optimizer is not None:
            optimizer.load_state_dict(state["optimizer"])
        if ema is not None:
            ema.load_state_dict(state["ema"], model)
        if rng is not None:
            _restore_rng_state(rng, generator)
    except Exception:
        try:
            model.load_state_dict(model_backup)
            if optimizer is not None and optimizer_backup is not None:
                optimizer.load_state_dict(optimizer_backup)
        finally:
            if ema is not None and ema_backup is not None:
                ema.decay, ema.shadow = ema_backup
            if rng_backup is not None:
                _restore_rng_state(rng_backup, generator)
        raise
    return result


def _validate_rng_state(loaded: Any, generator: torch.Generator | None) -> dict[str, Any]:
    if (
        not isinstance(loaded, dict)
        or not {"torch", "python", "cuda", "generator"} <= loaded.keys()
    ):
        raise ValueError("RNG state must contain torch, python, cuda and generator")
    if not isinstance(loaded["torch"], torch.Tensor):
        raise ValueError("Torch RNG state must be a tensor")
    torch.Generator().set_state(loaded["torch"].cpu())
    random.Random().setstate(loaded["python"])
    cuda = loaded["cuda"]
    if not isinstance(cuda, list) or not all(isinstance(value, torch.Tensor) for value in cuda):
        raise ValueError("CUDA RNG states must be a list of tensors")
    if cuda:
        if not torch.cuda.is_available() or len(cuda) != torch.cuda.device_count():
            raise ValueError("saved CUDA RNG states require matching available CUDA devices")
        for index, value in enumerate(cuda):
            torch.Generator(device=f"cuda:{index}").set_state(value.cpu())
    if generator is not None:
        if not isinstance(loaded["generator"], torch.Tensor):
            raise ValueError("checkpoint is missing requested generator state")
        generator.clone_state().set_state(loaded["generator"].cpu())
    return loaded


def _restore_rng_state(state: dict[str, Any], generator: torch.Generator | None) -> None:
    torch.set_rng_state(state["torch"].cpu())
    random.setstate(state["python"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    if generator is not None:
        generator.set_state(state["generator"].cpu())


# torch.nn.modules.module._EXTRA_STATE_KEY_SUFFIX (private in PyTorch; stable since 1.10).
_EXTRA_STATE_SUFFIX = "_extra_state"
_MOMENTUM_KEYS = frozenset({"exp_avg", "exp_avg_sq", "max_exp_avg_sq", "momentum_buffer"})


def _validate_model_state(model: nn.Module, loaded: Any) -> None:
    if not isinstance(loaded, dict):
        raise ValueError("model state must be a dict")
    expected = model.state_dict()
    missing = sorted(set(expected) - set(loaded))
    unexpected = sorted(set(loaded) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"model state keys do not match: missing={missing} unexpected={unexpected}"
        )
    for name, target in expected.items():
        value = loaded[name]
        if name.endswith(_EXTRA_STATE_SUFFIX):
            # PyTorch extra state (``get_extra_state``/``set_extra_state``) may be any
            # value the safe loader accepts. The owning module validates it in
            # ``set_extra_state``; a failure there is rolled back by the caller.
            if isinstance(target, torch.Tensor) != isinstance(value, torch.Tensor):
                raise ValueError(f"model extra state type mismatch: {name}")
            continue
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(target.shape):
            raise ValueError(f"model tensor shape mismatch: {name}")


def _validate_optimizer_state(optimizer: Optimizer, loaded: Any) -> None:
    if not isinstance(loaded, dict) or "param_groups" not in loaded or "state" not in loaded:
        raise ValueError("optimizer state must contain param_groups and state")
    groups = loaded["param_groups"]
    if not isinstance(groups, list) or not isinstance(loaded["state"], dict):
        raise ValueError("optimizer param_groups must be a list and state must be a dict")
    if len(groups) != len(optimizer.param_groups):
        raise ValueError("optimizer param group count does not match")
    for saved, current in zip(groups, optimizer.param_groups, strict=True):
        if not isinstance(saved, dict) or not isinstance(saved.get("params"), list):
            raise ValueError("optimizer param group must contain a parameter list")
        if len(saved["params"]) != len(current["params"]):
            raise ValueError("optimizer parameter count does not match")
        for saved_id, parameter in zip(saved["params"], current["params"], strict=True):
            entry = loaded["state"].get(saved_id, {})
            if not isinstance(entry, dict):
                raise ValueError("optimizer parameter state must be a dict")
            for key, value in entry.items():
                if key in _MOMENTUM_KEYS:
                    # Known per-parameter moments must match the parameter exactly,
                    # including zero-dimensional parameters. A scalar moment for a
                    # non-scalar parameter loads in PyTorch but fails at the next step.
                    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(
                        parameter.shape
                    ):
                        raise ValueError(f"optimizer state shape mismatch: {key}")
                elif key == "step":
                    _validate_step_counter(value)
                # Other keys (for example Adafactor's factored row_var/col_var/variance)
                # are optimizer-specific and are left to the optimizer to interpret.


def _validate_step_counter(value: Any) -> None:
    """Step counters must be finite, real, non-negative scalars.

    A Python number or a one-element tensor is accepted. Boolean and complex
    values are rejected: a Boolean tensor breaks the next optimizer step and a
    NaN or negative counter corrupts bias correction without raising.
    """
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("optimizer step counter must be a scalar")
        if value.dtype == torch.bool or value.is_complex():
            raise ValueError("optimizer step counter must be a real number")
        number = float(value.item())
    elif isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("optimizer step counter must be a real number")
    else:
        number = float(value)
    if not isfinite(number) or number < 0:
        raise ValueError("optimizer step counter must be finite and non-negative")
