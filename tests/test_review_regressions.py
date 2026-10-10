from dataclasses import replace
from pathlib import Path
from types import MethodType

import pytest
import torch
from torch import nn

from adalift.adaptive.policies import ResidualHaltingPolicy
from adalift.config import ModelConfig, TrainConfig
from adalift.flow.losses import (
    build_trajectory_targets,
    lift_trajectory_loss,
    sample_depth_coordinates,
)
from adalift.flow.sampling import adaptive_euler_sample, euler_sample
from adalift.models.dense import DenseDiT
from adalift.models.lift import LiFT
from adalift.training.checkpoint import load_checkpoint, save_checkpoint
from adalift.training.ema import EMA


def config() -> ModelConfig:
    return ModelConfig(
        in_channels=1,
        input_size=4,
        patch_size=2,
        hidden_size=16,
        depth=2,
        prelude_depth=1,
        recurrent_depth=1,
        num_heads=2,
        num_classes=3,
        time_embed_freq_dim=8,
        depth_embed_freq_dim=8,
    )


def inputs(batch: int = 3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.randn(batch, 1, 4, 4), torch.full((batch,), 0.5), torch.arange(batch) % 3


def test_heterogeneous_endpoints_and_real_accounting(monkeypatch: pytest.MonkeyPatch) -> None:
    model = LiFT(config()).eval()
    calls: list[tuple[list[int], list[float]]] = []
    endpoint_values = torch.tensor([101.0, 202.0, 303.0])

    def condition(
        self: LiFT, t: torch.Tensor, labels: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        return labels.float()[:, None]

    def prelude(
        self: LiFT, x: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        return torch.zeros(x.shape[0], 1, 1), (1, 1)

    def step(
        self: LiFT,
        h: torch.Tensor,
        h0: torch.Tensor,
        c: torch.Tensor,
        depth: torch.Tensor,
        grid: tuple[int, int],
    ) -> torch.Tensor:
        labels = c[:, 0].long()
        calls.append((labels.tolist(), depth.tolist()))
        return torch.where((depth == 1)[:, None, None], endpoint_values[labels, None, None], h + 1)

    class Head(nn.Module):
        def forward(self, h: torch.Tensor, c: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
            return h[:, :, :, None]

    def halt(
        self: ResidualHaltingPolicy, current: torch.Tensor, previous: torch.Tensor, loop_index: int
    ) -> torch.Tensor:
        # Active count distinguishes the first and second calls without tying
        # expected outputs to the implementation's indexing operations.
        if loop_index == 1:
            return torch.tensor([True, False, False])
        if loop_index == 2:
            return torch.tensor([True, False])
        return torch.zeros(current.shape[0], dtype=torch.bool)

    monkeypatch.setattr(model, "_condition", MethodType(condition, model))
    monkeypatch.setattr(model, "encode_prelude", MethodType(prelude, model))
    monkeypatch.setattr(model, "recurrent_step", MethodType(step, model))
    monkeypatch.setattr(model, "head", Head())
    monkeypatch.setattr(ResidualHaltingPolicy, "should_halt", halt)
    result = model.forward_adaptive(*inputs(), max_loops=4, min_loops=1)
    assert calls == [
        ([0, 1, 2], [0.25] * 3),
        ([0], [1.0]),
        ([1, 2], [0.5] * 2),
        ([1], [1.0]),
        ([2], [0.75]),
        ([2], [1.0]),
    ]
    torch.testing.assert_close(result.prediction.flatten(), endpoint_values)
    assert result.loops_used.tolist() == [2, 3, 4]
    assert result.total_core_evaluations == sum(len(rows) for rows, _ in calls) == 9
    assert result.mean_loops == 3
    assert torch.isnan(result.residuals[1][0])
    assert torch.isnan(result.residuals[2][:2]).all()


@pytest.mark.parametrize("budget,minimum", [(1, 1), (2, 1), (4, 4), (5, 1)])
def test_endpoint_budget_boundaries(budget: int, minimum: int) -> None:
    model = LiFT(config()).eval()
    args = inputs()
    result = model.forward_adaptive(
        *args, max_loops=budget, min_loops=minimum, residual_threshold=1e9
    )
    expected = min(budget, minimum + 1)
    assert result.loops_used.tolist() == [expected] * 3
    assert result.total_core_evaluations == 3 * expected
    assert torch.isfinite(result.prediction).all()


def test_no_halt_matches_fixed_and_preserves_order() -> None:
    torch.manual_seed(91)
    model = LiFT(config()).eval()
    with torch.no_grad():
        model.head.proj.weight.normal_()
    args = inputs()
    adaptive = model.forward_adaptive(*args, max_loops=4, min_loops=4)
    torch.testing.assert_close(adaptive.prediction, model(*args, num_loops=4))
    perm = torch.tensor([2, 0, 1])
    reordered = model.forward_adaptive(*(arg[perm] for arg in args), max_loops=4, min_loops=4)
    torch.testing.assert_close(reordered.prediction, adaptive.prediction[perm])
    assert adaptive.loops_used.tolist() == [4, 4, 4]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("model_type", [LiFT, DenseDiT])
def test_low_precision_models(dtype: torch.dtype, model_type: type[nn.Module]) -> None:
    model = model_type(config()).to(dtype=dtype).eval()
    x, t, y = inputs()
    output = model(x.to(dtype), t, y)
    assert output.dtype == dtype
    assert torch.isfinite(output).all()


def test_sampler_common_noise_and_modes() -> None:
    model = LiFT(replace(config(), class_dropout_prob=0.5))
    with torch.no_grad():
        model.head.proj.weight.normal_()
    model.label_embed.eval()  # A mixed module-mode state must also be restored.
    modes = [module.training for module in model.modules()]
    x, _, labels = inputs()
    initial = x.clone()
    fixed = euler_sample(model, tuple(x.shape), labels, num_steps=2, num_loops=3, initial_noise=x)
    adaptive, trace = adaptive_euler_sample(
        model, tuple(x.shape), labels, num_steps=2, max_loops=3, min_loops=3, initial_noise=x
    )
    torch.testing.assert_close(fixed, adaptive)
    torch.testing.assert_close(x, initial)
    assert trace.tolist() == [[3] * 3] * 2
    assert [module.training for module in model.modules()] == modes
    again = euler_sample(model, tuple(x.shape), labels, num_steps=2, num_loops=3, initial_noise=x)
    torch.testing.assert_close(fixed, again)


def test_sampler_restores_modes_on_failure() -> None:
    model = LiFT(config()).train()
    x, _, labels = inputs()
    with pytest.raises(ValueError):
        euler_sample(model, tuple(x.shape), labels, initial_noise=x, num_loops=0)
    assert model.training and model.label_embed.training
    with pytest.raises(ValueError):
        adaptive_euler_sample(model, tuple(x.shape), labels, num_steps=0)


def test_dropout_generator_and_detached_loss_targets() -> None:
    model = LiFT(replace(config(), class_dropout_prob=0.5)).train()
    with torch.no_grad():
        model.head.proj.weight.normal_()
    x, _, y = inputs()
    first = lift_trajectory_loss(
        model, x, y, k_train=2, generator=torch.Generator().manual_seed(42)
    )
    torch.rand(100)  # Change the global RNG independently of the supplied generator.
    second = lift_trajectory_loss(
        model, x, y, k_train=2, generator=torch.Generator().manual_seed(42)
    )
    torch.testing.assert_close(first.loss, second.loss)
    torch.testing.assert_close(first.depth_coordinates, second.depth_coordinates)
    anchor = torch.randn(3, 1, 4, 4, requires_grad=True)
    targets = build_trajectory_targets(anchor, x, torch.tensor([[0.25, 1.0]] * 3))
    assert not targets.requires_grad
    torch.testing.assert_close(targets[:, 0], 0.75 * anchor.detach() + 0.25 * x)
    torch.testing.assert_close(targets[:, -1], x)


def test_paper_depth_precision_and_spacing() -> None:
    depths = sample_depth_coordinates(
        32,
        64,
        device=torch.device("cpu"),
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(5),
    )
    assert depths.dtype == torch.float32
    assert bool((depths[:, 0] >= 1e-6).all())
    assert bool((depths[:, 1:] > depths[:, :-1]).all())
    assert bool((depths[:, -1] == 1).all())
    with pytest.raises(ValueError):
        sample_depth_coordinates(2, 4, device=torch.device("cpu"), dtype=torch.float16)


class UnsupportedCheckpoint:
    pass


def test_checkpoint_rejects_pickle_globals(tmp_path: Path) -> None:
    path = tmp_path / "unsupported.pt"
    torch.save({"model": UnsupportedCheckpoint()}, path)
    import pickle

    with pytest.raises(pickle.UnpicklingError):
        load_checkpoint(path, model=nn.Linear(2, 2))


def test_checkpoint_requires_requested_state_and_plain_metadata(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    path = tmp_path / "state.pt"
    save_checkpoint(path, model=model)
    with pytest.raises(ValueError, match="optimizer"):
        load_checkpoint(path, model=model, optimizer=torch.optim.AdamW(model.parameters()))
    with pytest.raises(ValueError, match="ema"):
        load_checkpoint(path, model=model, ema=EMA(model))
    with pytest.raises(TypeError):
        save_checkpoint(path, model=model, metadata={"bad": UnsupportedCheckpoint()})


def test_full_checkpoint_state_and_rng_resume(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    ema = EMA(model, 0.7)
    ema.update(model)
    generator = torch.Generator().manual_seed(3)
    path = tmp_path / "state.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        ema=ema,
        step=1,
        metadata={"model_config": {"width": 2}},
        capture_rng=True,
        generator=generator,
    )
    expected_global = torch.rand(3)
    expected_generator = torch.rand(3, generator=generator)
    loaded_model = nn.Linear(2, 2)
    loaded_optimizer = torch.optim.AdamW(loaded_model.parameters(), lr=0.9)
    loaded_ema = EMA(loaded_model, 0.1)
    loaded_generator = torch.Generator()
    load_checkpoint(
        path,
        model=loaded_model,
        optimizer=loaded_optimizer,
        ema=loaded_ema,
        restore_rng=True,
        generator=loaded_generator,
    )
    torch.testing.assert_close(torch.rand(3), expected_global)
    torch.testing.assert_close(torch.rand(3, generator=loaded_generator), expected_generator)
    assert loaded_optimizer.param_groups[0]["lr"] == 0.03
    for name, value in model.state_dict().items():
        torch.testing.assert_close(loaded_model.state_dict()[name], value)
        torch.testing.assert_close(loaded_ema.shadow[name], ema.shadow[name])
    for old, new in zip(optimizer.state.values(), loaded_optimizer.state.values(), strict=True):
        for key in old:
            torch.testing.assert_close(new[key], old[key])
    assert loaded_ema.decay == 0.7


def test_ema_casts_and_validates_loaded_state() -> None:
    source = nn.Linear(2, 2)
    target = nn.Linear(2, 2).double()
    ema = EMA(target)
    ema.load_state_dict(EMA(source).state_dict(), target)
    assert all(value.dtype == torch.float64 for value in ema.shadow.values())
    ema.update(target)
    with pytest.raises(ValueError):
        ema.load_state_dict({"decay": 0.9, "shadow": {}})


@pytest.mark.parametrize(
    "field,value",
    [
        ("patch_size", 0),
        ("num_heads", 0),
        ("patch_size", -2),
        ("class_dropout_prob", 1.5),
        ("mlp_ratio", 0),
        ("in_channels", 0),
        ("time_embed_freq_dim", 1),
    ],
)
def test_model_config_rejects_invalid_values(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        replace(config(), **{field: value}).validate()


@pytest.mark.parametrize(
    "field,value",
    [
        ("prelude_loss_weight", float("nan")),
        ("weight_decay", -1),
        ("grad_clip_norm", -1),
        ("seed", -1),
    ],
)
def test_training_config_rejects_invalid_values(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        replace(TrainConfig(), **{field: value}).validate()


def test_rejects_broadcast_condition_and_invalid_threshold() -> None:
    model = LiFT(config())
    x, t, y = inputs()
    with pytest.raises(ValueError):
        model(x, t[:1], y)
    with pytest.raises(ValueError):
        model.forward_adaptive(x, t, y, residual_threshold=float("nan"))


def test_loss_value_matches_independent_formula() -> None:
    model = LiFT(config()).eval()
    x1, _, labels = inputs()
    generator = torch.Generator().manual_seed(9)
    x0 = torch.randn(x1.shape, generator=generator)
    t = torch.rand(x1.shape[0], generator=generator)
    interior = torch.rand(x1.shape[0], 1, generator=generator, dtype=torch.float64)
    depth = ((1 - 2e-6) * interior + 1e-6).float()
    depths = torch.cat([depth, torch.ones(x1.shape[0], 1)], dim=1)
    xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1
    trajectory = model.forward_trajectory(xt, t, labels, depths)
    target = x1 - x0
    losses = []
    for index, prediction in enumerate(trajectory.predictions):
        s = depths[:, index, None, None, None]
        path_target = (1 - s) * trajectory.prelude_prediction.detach() + s * target
        losses.append((prediction - path_target).square().mean())
    expected = (
        torch.stack(losses).mean() + 0.01 * (trajectory.prelude_prediction - target).square().mean()
    )
    actual = lift_trajectory_loss(
        model, x1, labels, k_train=2, generator=torch.Generator().manual_seed(9)
    )
    torch.testing.assert_close(actual.loss, expected)


def test_fixed_default_inference_reads_only_final_head() -> None:
    model = LiFT(config()).eval()
    calls: list[int] = []
    hook = model.head.register_forward_hook(lambda *args: calls.append(1))
    try:
        model(*inputs(), num_loops=5)
        assert len(calls) == 1
    finally:
        hook.remove()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_checkpoint_ema_moves_cpu_state_to_cuda(tmp_path: Path) -> None:
    cpu_model = nn.Linear(2, 2)
    path = tmp_path / "gpu.pt"
    save_checkpoint(path, model=cpu_model, ema=EMA(cpu_model))
    model = nn.Linear(2, 2).cuda()
    ema = EMA(model)
    load_checkpoint(path, model=model, ema=ema, map_location="cpu")
    assert all(value.device.type == "cuda" for value in ema.shadow.values())
    ema.update(model)
