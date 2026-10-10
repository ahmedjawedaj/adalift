from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from adalift.adaptive.policies import ResidualHaltingPolicy
from adalift.config import ModelConfig
from adalift.models.blocks import DiTBlock, FinalLayer
from adalift.models.embeddings import (
    LabelEmbedding,
    PatchEmbed2d,
    ScalarConditionEmbedding,
    validate_condition_inputs,
)


@dataclass(slots=True)
class LiFTTrajectory:
    prelude_prediction: torch.Tensor
    predictions: list[torch.Tensor]
    depth_coordinates: torch.Tensor


@dataclass(slots=True)
class AdaptiveLiFTOutput:
    prediction: torch.Tensor
    loops_used: torch.Tensor
    residuals: list[torch.Tensor]

    @property
    def total_core_evaluations(self) -> int:
        return int(self.loops_used.sum().item())

    @property
    def mean_loops(self) -> float:
        return float(self.loops_used.float().mean().item())


class LiFT(nn.Module):
    """Loop Flow Transformer.

    The implementation follows the paper-level decomposition:
    patch embedding + prelude, a shared recurrent core, input-state reinjection at each
    recurrent boundary, continuous depth conditioning inside the core, and a shared
    prediction-only head.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.patch_embed = PatchEmbed2d(config.in_channels, config.hidden_size, config.patch_size)
        self.time_embed = ScalarConditionEmbedding(config.hidden_size, config.time_embed_freq_dim)
        self.label_embed = LabelEmbedding(
            config.num_classes,
            config.hidden_size,
            config.class_dropout_prob,
        )
        self.depth_embed = ScalarConditionEmbedding(config.hidden_size, config.depth_embed_freq_dim)

        def make_block() -> DiTBlock:
            return DiTBlock(
                config.hidden_size,
                config.num_heads,
                config.mlp_ratio,
                qk_norm=config.qk_norm,
                rope_theta=config.rope_theta,
            )

        self.prelude = nn.ModuleList([make_block() for _ in range(config.prelude_depth)])
        self.recurrent_core = nn.ModuleList([make_block() for _ in range(config.recurrent_depth)])
        self.boundary_norm = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.head = FinalLayer(config.hidden_size, config.in_channels, config.patch_size)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight.view(module.weight.shape[0], -1))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(init)
        # Re-zero adaLN gates and final projection after generic initialization.
        for block in [*self.prelude, *self.recurrent_core]:
            assert isinstance(block, DiTBlock)
            block._reset_parameters()
        self.head._reset_parameters()

    def _condition(
        self, t: torch.Tensor, y: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        condition: torch.Tensor = self.time_embed(t) + self.label_embed(y, generator=generator)
        return condition

    def encode_prelude(
        self, x: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        h, grid = self.patch_embed(x)
        for block in self.prelude:
            assert isinstance(block, DiTBlock)
            h = block(h, condition, grid)
        return h, grid

    def recurrent_step(
        self,
        h: torch.Tensor,
        h0: torch.Tensor,
        condition: torch.Tensor,
        depth: torch.Tensor,
        grid_size: tuple[int, int],
    ) -> torch.Tensor:
        if depth.ndim != 1 or depth.shape[0] != h.shape[0]:
            raise ValueError("depth must have shape [batch]")
        h = self.boundary_norm(h) + h0
        core_condition = condition + self.depth_embed(depth)
        for block in self.recurrent_core:
            assert isinstance(block, DiTBlock)
            h = block(h, core_condition, grid_size)
        return h

    def forward_trajectory(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        depth_coordinates: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> LiFTTrajectory:
        validate_condition_inputs(x, t, y, self.config.num_classes)
        if depth_coordinates.ndim != 2:
            raise ValueError("depth_coordinates must have shape [batch, loops]")
        if depth_coordinates.shape[0] != x.shape[0]:
            raise ValueError("depth coordinate batch size mismatch")
        if depth_coordinates.shape[1] < 1 or depth_coordinates.device != x.device:
            raise ValueError("depth coordinates must be nonempty and on the input device")
        if not bool(torch.isfinite(depth_coordinates).all()) or not bool(
            ((depth_coordinates >= 0) & (depth_coordinates <= 1)).all()
        ):
            raise ValueError("depth coordinates must be finite and in [0, 1]")
        if not bool((depth_coordinates[:, 1:] >= depth_coordinates[:, :-1]).all()):
            raise ValueError("depth coordinates must be sorted")
        condition = self._condition(t, y, generator)
        h0, grid = self.encode_prelude(x, condition)
        prelude_prediction = self.head(h0, condition, grid)
        h = h0
        predictions: list[torch.Tensor] = []
        for idx in range(depth_coordinates.shape[1]):
            h = self.recurrent_step(h, h0, condition, depth_coordinates[:, idx], grid)
            predictions.append(self.head(h, condition, grid))
        return LiFTTrajectory(prelude_prediction, predictions, depth_coordinates)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        *,
        num_loops: int = 1,
        depth_coordinates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if num_loops < 1:
            raise ValueError("num_loops must be >= 1")
        validate_condition_inputs(x, t, y, self.config.num_classes)
        if depth_coordinates is None:
            condition = self._condition(t, y)
            h0, grid = self.encode_prelude(x, condition)
            h = h0
            for index in range(1, num_loops + 1):
                depth = torch.full((x.shape[0],), index / num_loops, device=x.device)
                h = self.recurrent_step(h, h0, condition, depth, grid)
            result: torch.Tensor = self.head(h, condition, grid)
            return result
        else:
            if depth_coordinates.shape != (x.shape[0], num_loops):
                raise ValueError("depth_coordinates shape must be [batch, num_loops]")
            if not bool((depth_coordinates[:, -1] == 1).all()):
                raise ValueError("forward requires an s=1 endpoint")
        trajectory = self.forward_trajectory(x, t, y, depth_coordinates)
        return trajectory.predictions[-1]

    @torch.no_grad()
    def forward_adaptive(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        *,
        max_loops: int = 16,
        min_loops: int = 2,
        residual_threshold: float = 0.01,
        eps: float = 1e-8,
    ) -> AdaptiveLiFTOutput:
        """Experimental residual halting with an endpoint pass and active compaction.

        max_loops caps actual core evaluations, including the final s=1 pass.
        min_loops specifies the first intermediate residual check. Inactive rows
        are NaN in residuals. Evaluation counts do not measure FLOPs or latency.
        """
        policy = ResidualHaltingPolicy(residual_threshold, min_loops, max_loops, eps)
        policy.validate()
        validate_condition_inputs(x, t, y, self.config.num_classes)
        condition = self._condition(t, y)
        h0, grid = self.encode_prelude(x, condition)
        h = h0
        previous = self.head(h0, condition, grid)
        prediction = torch.empty_like(previous)
        active = torch.arange(x.shape[0], device=x.device)
        loops_used = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        residual_history: list[torch.Tensor] = []

        for loop_idx in range(1, max_loops + 1):
            depth = torch.full((active.numel(),), loop_idx / max_loops, device=x.device)
            candidate_h = self.recurrent_step(h, h0[active], condition[active], depth, grid)
            loops_used[active] += 1
            candidate = self.head(candidate_h, condition[active], grid)
            residual = policy.residual(candidate, previous)
            recorded = torch.full((x.shape[0],), float("nan"), device=x.device)
            recorded[active] = residual
            residual_history.append(recorded)
            if loop_idx == max_loops:
                prediction[active] = candidate
                break

            exiting = policy.should_halt(candidate, previous, loop_idx)
            if bool(exiting.any()):
                exited_rows = active[exiting]
                endpoint_h = self.recurrent_step(
                    candidate_h[exiting],
                    h0[exited_rows],
                    condition[exited_rows],
                    torch.ones(exited_rows.numel(), device=x.device),
                    grid,
                )
                loops_used[exited_rows] += 1
                prediction[exited_rows] = self.head(endpoint_h, condition[exited_rows], grid)
            keep = ~exiting
            active = active[keep]
            if active.numel() == 0:
                break
            h = candidate_h[keep]
            previous = candidate[keep]

        return AdaptiveLiFTOutput(prediction, loops_used, residual_history)
