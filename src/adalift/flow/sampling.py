from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch

from adalift.models.embeddings import validate_condition_inputs
from adalift.models.lift import LiFT


@contextmanager
def _evaluation_mode(model: LiFT) -> Iterator[None]:
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training


def _initial_state(
    model: LiFT,
    shape: tuple[int, int, int, int],
    labels: torch.Tensor,
    num_steps: int,
    generator: torch.Generator | None,
    initial_noise: torch.Tensor | None,
) -> torch.Tensor:
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if len(shape) != 4 or any(size < 1 for size in shape):
        raise ValueError("shape must contain four positive dimensions")
    parameter = next(model.parameters())
    if labels.device != parameter.device:
        raise ValueError("labels and model must share a device")
    if initial_noise is None:
        x = torch.randn(shape, device=parameter.device, dtype=parameter.dtype, generator=generator)
    else:
        if tuple(initial_noise.shape) != shape:
            raise ValueError("initial_noise shape mismatch")
        if initial_noise.device != parameter.device or initial_noise.dtype != parameter.dtype:
            raise ValueError("initial_noise must match model device and dtype")
        if not bool(torch.isfinite(initial_noise).all()):
            raise ValueError("initial_noise must be finite")
        x = initial_noise.clone()
    validate_condition_inputs(
        x, torch.zeros(shape[0], device=x.device), labels, model.config.num_classes
    )
    return x


@torch.no_grad()
def euler_sample(
    model: LiFT,
    shape: tuple[int, int, int, int],
    labels: torch.Tensor,
    *,
    num_steps: int = 50,
    num_loops: int = 4,
    generator: torch.Generator | None = None,
    initial_noise: torch.Tensor | None = None,
    callback: Callable[[int, torch.Tensor], None] | None = None,
) -> torch.Tensor:
    """Euler integrate in evaluation mode and restore the caller's module modes."""
    x = _initial_state(model, shape, labels, num_steps, generator, initial_noise)
    dt = 1.0 / num_steps
    with _evaluation_mode(model):
        for step in range(num_steps):
            t = torch.full((shape[0],), step / num_steps, device=x.device)
            velocity = model(x, t, labels, num_loops=num_loops)
            x = x + dt * velocity
            if callback is not None:
                callback(step, x.clone())
    return x


@torch.no_grad()
def adaptive_euler_sample(
    model: LiFT,
    shape: tuple[int, int, int, int],
    labels: torch.Tensor,
    *,
    num_steps: int = 50,
    max_loops: int = 16,
    min_loops: int = 2,
    residual_threshold: float = 0.01,
    generator: torch.Generator | None = None,
    initial_noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Euler sampling with residual-based adaptive inner depth.

    Returns samples and ``[num_steps, batch]`` actual core-evaluation counts,
    including endpoint passes. Counts exclude prelude, heads and overhead;
    they do not measure FLOPs or wall-clock savings. Modes are restored.
    """
    x = _initial_state(model, shape, labels, num_steps, generator, initial_noise)
    dt = 1.0 / num_steps
    loop_trace: list[torch.Tensor] = []
    with _evaluation_mode(model):
        for step in range(num_steps):
            t = torch.full((shape[0],), step / num_steps, device=x.device)
            result = model.forward_adaptive(
                x,
                t,
                labels,
                max_loops=max_loops,
                min_loops=min_loops,
                residual_threshold=residual_threshold,
            )
            x = x + dt * result.prediction
            loop_trace.append(result.loops_used)
    return x, torch.stack(loop_trace)
