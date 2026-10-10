from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True, slots=True)
class ModelProfile:
    total_parameters: int
    trainable_parameters: int


def profile_parameters(model: nn.Module) -> ModelProfile:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return ModelProfile(total, trainable)
