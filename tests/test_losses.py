import torch

from adalift.config import ModelConfig
from adalift.flow.losses import (
    build_trajectory_targets,
    lift_trajectory_loss,
    sample_depth_coordinates,
)
from adalift.models.lift import LiFT


def test_depth_coordinates_are_sorted_and_end_at_one() -> None:
    depths = sample_depth_coordinates(
        16,
        4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(7),
    )
    assert depths.shape == (16, 4)
    assert torch.all(depths[:, 1:] >= depths[:, :-1])
    assert torch.all(depths[:, -1] == 1)


def test_trajectory_target_endpoints() -> None:
    anchor = torch.zeros(2, 1, 2, 2)
    target = torch.ones_like(anchor)
    depths = torch.tensor([[0.25, 1.0], [0.75, 1.0]])
    out = build_trajectory_targets(anchor, target, depths)
    torch.testing.assert_close(out[0, 0], torch.full_like(anchor[0], 0.25))
    torch.testing.assert_close(out[:, -1], target)


def test_loss_is_finite_and_backpropagates() -> None:
    cfg = ModelConfig(
        in_channels=4,
        input_size=8,
        patch_size=2,
        hidden_size=64,
        depth=4,
        prelude_depth=1,
        recurrent_depth=1,
        num_heads=4,
        num_classes=5,
        depth_embed_freq_dim=32,
        time_embed_freq_dim=32,
    )
    model = LiFT(cfg)
    x1 = torch.randn(2, 4, 8, 8)
    y = torch.tensor([0, 1])
    result = lift_trajectory_loss(model, x1, y, k_train=2)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
