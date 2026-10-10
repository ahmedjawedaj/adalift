import pytest
import torch

from adalift.config import ModelConfig
from adalift.flow.losses import build_trajectory_targets, sample_depth_coordinates
from adalift.models.embeddings import PatchEmbed2d, sinusoidal_embedding
from adalift.models.lift import LiFT
from adalift.models.rope import apply_2d_rope


def test_common_error_paths() -> None:
    with pytest.raises(ValueError):
        sinusoidal_embedding(torch.zeros(1, 1), 8)
    with pytest.raises(ValueError):
        PatchEmbed2d(4, 8, 2)(torch.zeros(1, 4, 8))
    with pytest.raises(ValueError):
        sample_depth_coordinates(1, 0, device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(ValueError):
        build_trajectory_targets(torch.zeros(1, 2), torch.zeros(1, 3), torch.ones(1, 1))
    with pytest.raises(ValueError):
        build_trajectory_targets(torch.zeros(1, 2), torch.zeros(1, 2), torch.ones(2, 1))


def test_rope_error_paths() -> None:
    q = torch.randn(1, 2, 4, 8)
    with pytest.raises(ValueError):
        apply_2d_rope(q, torch.randn(1, 2, 4, 12), (2, 2))
    with pytest.raises(ValueError):
        apply_2d_rope(torch.randn(1, 2, 4, 6), torch.randn(1, 2, 4, 6), (2, 2))
    with pytest.raises(ValueError):
        apply_2d_rope(torch.randn(1, 2, 5, 8), torch.randn(1, 2, 5, 8), (2, 2))


def test_lift_input_validation() -> None:
    model = LiFT(
        ModelConfig(
            in_channels=4,
            input_size=8,
            patch_size=2,
            hidden_size=64,
            depth=4,
            prelude_depth=1,
            recurrent_depth=1,
            num_heads=4,
            num_classes=3,
            depth_embed_freq_dim=32,
            time_embed_freq_dim=32,
        )
    )
    x = torch.randn(1, 4, 8, 8)
    t = torch.tensor([0.5])
    y = torch.tensor([1])
    with pytest.raises(ValueError):
        model(x, t, y, num_loops=0)
    with pytest.raises(ValueError):
        model(x, t, y, num_loops=2, depth_coordinates=torch.ones(1, 3))
    with pytest.raises(ValueError):
        model.forward_trajectory(x, t, y, torch.ones(2))
    with pytest.raises(ValueError):
        model.forward_adaptive(x, t, y, max_loops=2, min_loops=3)
