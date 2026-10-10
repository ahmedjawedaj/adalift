import torch

from adalift.config import ModelConfig
from adalift.evaluation import profile_parameters
from adalift.models import DenseDiT, LiFT


def config() -> ModelConfig:
    return ModelConfig(
        in_channels=4,
        input_size=8,
        patch_size=2,
        hidden_size=64,
        depth=4,
        prelude_depth=1,
        recurrent_depth=1,
        num_heads=4,
        num_classes=4,
        depth_embed_freq_dim=32,
        time_embed_freq_dim=32,
    )


def test_dense_baseline_shape_and_parameter_gap() -> None:
    cfg = config()
    dense = DenseDiT(cfg)
    lift = LiFT(cfg)
    x = torch.randn(2, 4, 8, 8)
    t = torch.rand(2)
    y = torch.tensor([0, 1])
    assert dense(x, t, y).shape == x.shape
    assert profile_parameters(dense).total_parameters > profile_parameters(lift).total_parameters
