import torch

from adalift.config import ModelConfig
from adalift.flow.sampling import adaptive_euler_sample, euler_sample
from adalift.models.lift import LiFT


def model() -> LiFT:
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
            num_classes=4,
            depth_embed_freq_dim=32,
            time_embed_freq_dim=32,
        )
    )


def test_euler_with_initial_noise_and_callback() -> None:
    m = model()
    labels = torch.tensor([1])
    initial = torch.randn(1, 4, 8, 8)
    calls: list[int] = []
    out = euler_sample(
        m,
        tuple(initial.shape),
        labels,
        num_steps=2,
        num_loops=1,
        initial_noise=initial,
        callback=lambda step, _: calls.append(step),
    )
    assert out.shape == initial.shape
    assert calls == [0, 1]


def test_adaptive_sampler_returns_trace() -> None:
    m = model()
    labels = torch.tensor([1, 2])
    out, trace = adaptive_euler_sample(
        m,
        (2, 4, 8, 8),
        labels,
        num_steps=2,
        max_loops=3,
        min_loops=1,
        residual_threshold=1e9,
    )
    assert out.shape == (2, 4, 8, 8)
    assert trace.shape == (2, 2)
    assert torch.all(trace == 1)
