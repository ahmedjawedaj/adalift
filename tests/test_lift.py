import torch

from adalift.config import ModelConfig
from adalift.models.lift import LiFT


def tiny_model() -> LiFT:
    return LiFT(
        ModelConfig(
            in_channels=4,
            input_size=8,
            patch_size=2,
            hidden_size=64,
            depth=4,
            prelude_depth=1,
            recurrent_depth=1,
            num_heads=4,
            num_classes=10,
            depth_embed_freq_dim=32,
            time_embed_freq_dim=32,
        )
    )


def test_fixed_depth_forward_shape() -> None:
    model = tiny_model()
    x = torch.randn(2, 4, 8, 8)
    t = torch.tensor([0.2, 0.7])
    y = torch.tensor([1, 2])
    out = model(x, t, y, num_loops=4)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_trajectory_has_one_prediction_per_depth() -> None:
    model = tiny_model()
    x = torch.randn(2, 4, 8, 8)
    t = torch.tensor([0.2, 0.7])
    y = torch.tensor([1, 2])
    depths = torch.tensor([[0.25, 1.0], [0.4, 1.0]])
    trajectory = model.forward_trajectory(x, t, y, depths)
    assert trajectory.prelude_prediction.shape == x.shape
    assert len(trajectory.predictions) == 2
    assert all(pred.shape == x.shape for pred in trajectory.predictions)


def test_depth_extrapolation_is_supported_by_api() -> None:
    model = tiny_model()
    x = torch.randn(1, 4, 8, 8)
    t = torch.tensor([0.5])
    y = torch.tensor([3])
    # The API intentionally allows K_inf > the training loop count.
    out = model(x, t, y, num_loops=8)
    assert out.shape == x.shape


def test_adaptive_forward_returns_loop_counts() -> None:
    model = tiny_model()
    x = torch.randn(2, 4, 8, 8)
    t = torch.tensor([0.3, 0.3])
    y = torch.tensor([1, 2])
    result = model.forward_adaptive(x, t, y, max_loops=4, min_loops=1, residual_threshold=1e9)
    assert result.prediction.shape == x.shape
    assert result.loops_used.shape == (2,)
    assert torch.equal(result.loops_used, torch.full_like(result.loops_used, 2))
