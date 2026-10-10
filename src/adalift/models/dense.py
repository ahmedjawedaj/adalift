from __future__ import annotations

import torch
from torch import nn

from adalift.config import ModelConfig
from adalift.models.blocks import DiTBlock, FinalLayer
from adalift.models.embeddings import (
    LabelEmbedding,
    PatchEmbed2d,
    ScalarConditionEmbedding,
    validate_condition_inputs,
)


class DenseDiT(nn.Module):
    """Dense DiT baseline using the same block design as :class:`adalift.models.lift.LiFT`."""

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
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    config.hidden_size,
                    config.num_heads,
                    config.mlp_ratio,
                    qk_norm=config.qk_norm,
                    rope_theta=config.rope_theta,
                )
                for _ in range(config.depth)
            ]
        )
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
        for block in self.blocks:
            assert isinstance(block, DiTBlock)
            block._reset_parameters()
        self.head._reset_parameters()

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        validate_condition_inputs(x, t, y, self.config.num_classes)
        condition = self.time_embed(t) + self.label_embed(y)
        h, grid = self.patch_embed(x)
        for block in self.blocks:
            assert isinstance(block, DiTBlock)
            h = block(h, condition, grid)
        result: torch.Tensor = self.head(h, condition, grid)
        return result
