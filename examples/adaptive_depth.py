"""CPU API demonstration; untrained outputs are not a quality or speed benchmark."""

import torch

from adalift.config import ModelConfig
from adalift.flow import adaptive_euler_sample, euler_sample
from adalift.models import LiFT


def main() -> None:
    torch.manual_seed(42)
    torch.set_num_threads(2)
    config = ModelConfig(
        in_channels=4,
        input_size=16,
        patch_size=2,
        hidden_size=128,
        depth=6,
        prelude_depth=2,
        recurrent_depth=2,
        num_heads=4,
        num_classes=10,
    )
    model = LiFT(config).eval()
    noise = torch.randn(2, 4, 16, 16)
    labels = torch.tensor([1, 2])
    times = torch.tensor([0.25, 0.75])
    with torch.inference_mode():
        result = model.forward_adaptive(
            noise, times, labels, max_loops=8, min_loops=2, residual_threshold=0.01
        )
    print("Untrained CPU example: API bookkeeping, not quality or speed evidence.")
    print(f"Velocity shape: {tuple(result.prediction.shape)}")
    print(f"Core evaluations per sample (endpoint included): {result.loops_used.tolist()}")
    print(f"Total core evaluations: {result.total_core_evaluations}")
    print(f"Mean core evaluations: {result.mean_loops:.1f}")

    # min_loops == max_loops disables early halting for this equivalence check.
    fixed = euler_sample(
        model, tuple(noise.shape), labels, num_steps=3, num_loops=4, initial_noise=noise
    )
    adaptive, trace = adaptive_euler_sample(
        model,
        tuple(noise.shape),
        labels,
        num_steps=3,
        max_loops=4,
        min_loops=4,
        initial_noise=noise,
    )
    torch.testing.assert_close(adaptive, fixed)
    print(f"Sampling loop trace [flow steps, batch]: {trace.tolist()}")
    print("Fixed-depth and adaptive-without-halting samples agree on common initial noise.")
    print("The zero-initialized prediction head makes this a limited untrained check.")


if __name__ == "__main__":
    main()
