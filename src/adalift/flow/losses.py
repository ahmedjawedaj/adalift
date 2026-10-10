from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
import torch.nn.functional as F

from adalift.models.lift import LiFT


@dataclass(slots=True)
class LiFTLossOutput:
    loss: torch.Tensor
    trajectory_loss: torch.Tensor
    prelude_loss: torch.Tensor
    target_mse_by_depth: torch.Tensor
    depth_coordinates: torch.Tensor


def sample_depth_coordinates(
    batch_size: int,
    k_train: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample sorted random interior coordinates and append the endpoint ``s=1``."""
    if batch_size < 1 or not 1 <= k_train < 1_000_000:
        raise ValueError("batch_size must be positive and require 1 <= k_train < 1000000")
    if dtype != torch.float32:
        raise ValueError("training depth coordinates require float32 output precision")
    if k_train == 1:
        return torch.ones(batch_size, 1, device=device, dtype=dtype)
    interior = torch.rand(
        batch_size,
        k_train - 1,
        device=device,
        dtype=torch.float64,
        generator=generator,
    )
    interior, _ = interior.sort(dim=1)
    offsets = torch.arange(1, k_train, device=device, dtype=torch.float64) * 1e-6
    interior = ((1 - k_train * 1e-6) * interior + offsets).float()
    return torch.cat([interior, torch.ones(batch_size, 1, device=device)], dim=1)


def build_trajectory_targets(
    anchor: torch.Tensor,
    flow_target: torch.Tensor,
    depth_coordinates: torch.Tensor,
) -> torch.Tensor:
    """Interpolate in prediction space from stop-gradient anchor to flow target."""
    if anchor.shape != flow_target.shape:
        raise ValueError("anchor and flow_target shapes must match")
    if depth_coordinates.shape[0] != anchor.shape[0]:
        raise ValueError("depth coordinate batch mismatch")
    view_shape = (anchor.shape[0], depth_coordinates.shape[1]) + (1,) * (anchor.ndim - 1)
    s = depth_coordinates.reshape(view_shape)
    return (1.0 - s) * anchor.detach()[:, None] + s * flow_target[:, None]


def lift_trajectory_loss(
    model: LiFT,
    x1: torch.Tensor,
    y: torch.Tensor,
    *,
    k_train: int,
    prelude_loss_weight: float = 0.01,
    generator: torch.Generator | None = None,
) -> LiFTLossOutput:
    """Compute LiFT flow-matching trajectory supervision for one minibatch."""
    if not isfinite(prelude_loss_weight) or prelude_loss_weight < 0:
        raise ValueError("prelude_loss_weight must be finite and non-negative")
    x0 = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=generator)
    t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype, generator=generator)
    t_view = t.reshape(x1.shape[0], *([1] * (x1.ndim - 1)))
    xt = (1.0 - t_view) * x0 + t_view * x1
    flow_target = x1 - x0
    depths = sample_depth_coordinates(
        x1.shape[0],
        k_train,
        device=x1.device,
        dtype=torch.float32,
        generator=generator,
    )
    trajectory = model.forward_trajectory(xt, t, y, depths, generator=generator)
    target_stack = build_trajectory_targets(
        trajectory.prelude_prediction,
        flow_target,
        depths,
    )
    prediction_stack = torch.stack(trajectory.predictions, dim=1)
    reduce_dims = tuple(range(2, prediction_stack.ndim))
    mse_by_depth = (prediction_stack.float() - target_stack.float()).pow(2).mean(dim=reduce_dims)
    trajectory_loss = mse_by_depth.mean()
    prelude_loss = F.mse_loss(trajectory.prelude_prediction.float(), flow_target.float())
    loss = trajectory_loss + prelude_loss_weight * prelude_loss
    return LiFTLossOutput(
        loss, trajectory_loss, prelude_loss, mse_by_depth.detach(), depths.detach()
    )
