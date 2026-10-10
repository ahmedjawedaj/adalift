import torch

from adalift.config import ModelConfig
from adalift.flow.sampling import euler_sample
from adalift.models.lift import LiFT


def test_euler_sampler_shape() -> None:
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
            num_classes=4,
            depth_embed_freq_dim=32,
            time_embed_freq_dim=32,
        )
    )
    labels = torch.tensor([1, 2])
    out = euler_sample(model, (2, 4, 8, 8), labels, num_steps=2, num_loops=2)
    assert out.shape == (2, 4, 8, 8)
