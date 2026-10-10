from __future__ import annotations

from collections.abc import Callable

import torch

from adalift.models.lift import AdaptiveLiFTOutput, LiFT


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
    """Euler integrate the learned velocity from noise (t=0) to data (t=1)."""
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if initial_noise is None:
        x = torch.randn(
            shape,
            device=labels.device,
            dtype=next(model.parameters()).dtype,
            generator=generator,
        )
    else:
        if tuple(initial_noise.shape) != shape:
            raise ValueError("initial_noise shape mismatch")
        x = initial_noise.clone()
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t = torch.full((shape[0],), step / num_steps, device=x.device, dtype=x.dtype)
        velocity = model(x, t, labels, num_loops=num_loops)
        x = x + dt * velocity
        if callback is not None:
            callback(step, x)
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Euler sampling with residual-based adaptive inner depth.

    Returns the sample and a ``[num_steps, batch]`` tensor of loops used.
    """
    x = torch.randn(
        shape,
        device=labels.device,
        dtype=next(model.parameters()).dtype,
        generator=generator,
    )
    dt = 1.0 / num_steps
    loop_trace: list[torch.Tensor] = []
    for step in range(num_steps):
        t = torch.full((shape[0],), step / num_steps, device=x.device, dtype=x.dtype)
        result: AdaptiveLiFTOutput = model.forward_adaptive(
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
