import copy
import random
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from adalift.training import checkpoint
from adalift.training.ema import EMA


def assert_state_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, list | tuple):
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected, strict=True):
            assert_state_equal(value, reference)
    else:
        assert actual == expected


@pytest.mark.parametrize("field", ["torch", "python", "generator", "cuda", "step", "metadata"])
def test_invalid_checkpoint_state_does_not_mutate_callers(tmp_path: Path, field: str) -> None:
    source = nn.Linear(2, 2)
    target = nn.Linear(2, 2)
    with torch.no_grad():
        source.weight.fill_(7)
        target.weight.zero_()
    path = tmp_path / "state.pt"
    generator = torch.Generator().manual_seed(91)
    checkpoint.save_checkpoint(path, model=source, capture_rng=True, generator=generator)
    state = torch.load(path, weights_only=True)
    if field in {"step", "metadata"}:
        state[field] = "invalid step" if field == "step" else torch.ones(1)
    else:
        state["rng"][field] = None if field != "cuda" else [torch.zeros(1)]
    torch.save(state, path)
    before = copy.deepcopy(target.state_dict())
    torch_before = torch.get_rng_state()
    python_before = random.getstate()
    generator_before = generator.get_state()
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        checkpoint.load_checkpoint(path, model=target, restore_rng=True, generator=generator)
    assert_state_equal(target.state_dict(), before)
    assert_state_equal(torch.get_rng_state(), torch_before)
    assert random.getstate() == python_before
    assert_state_equal(generator.get_state(), generator_before)


def test_late_failure_rolls_back_model_optimizer_ema_and_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = nn.Linear(2, 2)
    source_optimizer = torch.optim.AdamW(source.parameters(), lr=0.03)
    source(torch.ones(1, 2)).sum().backward()
    source_optimizer.step()
    source_ema = EMA(source, 0.7)
    random.seed(3)
    path = tmp_path / "state.pt"
    checkpoint.save_checkpoint(
        path,
        model=source,
        optimizer=source_optimizer,
        ema=source_ema,
        capture_rng=True,
        generator=torch.Generator().manual_seed(5),
    )
    target = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(target.parameters(), lr=0.5)
    target(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    ema = EMA(target, 0.2)
    generator = torch.Generator().manual_seed(9)
    random.seed(99)
    before = copy.deepcopy((target.state_dict(), optimizer.state_dict(), ema.state_dict()))
    torch_before = torch.get_rng_state()
    python_before = random.getstate()
    generator_before = generator.get_state()
    restore = checkpoint._restore_rng_state
    calls = 0

    def fail_after_apply(state: dict[str, Any], supplied: torch.Generator | None) -> None:
        nonlocal calls
        restore(state, supplied)
        calls += 1
        if calls == 1:
            # Confirm application reached the last state before injecting failure.
            assert optimizer.param_groups[0]["lr"] == 0.03
            assert ema.decay == 0.7
            assert_state_equal(target.state_dict(), source.state_dict())
            assert random.getstate() != python_before
            raise RuntimeError("injected late application failure")

    monkeypatch.setattr(checkpoint, "_restore_rng_state", fail_after_apply)
    with pytest.raises(RuntimeError, match="injected late"):
        checkpoint.load_checkpoint(
            path, model=target, optimizer=optimizer, ema=ema, restore_rng=True, generator=generator
        )
    assert calls == 2
    assert_state_equal((target.state_dict(), optimizer.state_dict(), ema.state_dict()), before)
    assert_state_equal(torch.get_rng_state(), torch_before)
    assert random.getstate() == python_before
    assert_state_equal(generator.get_state(), generator_before)


def test_checkpoint_accepts_factored_optimizer_statistics(tmp_path: Path) -> None:
    source = nn.Linear(4, 3)
    source_optimizer = torch.optim.Adafactor(source.parameters())
    source(torch.ones(1, 4)).sum().backward()
    source_optimizer.step()
    path = tmp_path / "factored.pt"
    checkpoint.save_checkpoint(path, model=source, optimizer=source_optimizer)
    target = nn.Linear(4, 3)
    optimizer = torch.optim.Adafactor(target.parameters())
    checkpoint.load_checkpoint(path, model=target, optimizer=optimizer)
    assert_state_equal(target.state_dict(), source.state_dict())
    assert_state_equal(optimizer.state_dict(), source_optimizer.state_dict())


class ModuleWithExtraState(nn.Linear):
    """A module using PyTorch's extra-state hooks with plain, safely loadable values."""

    def __init__(self) -> None:
        super().__init__(2, 2)
        self.extra: dict[str, Any] = {"schema": 1, "tags": ["a", "b"], "scale": 0.5}

    def get_extra_state(self) -> dict[str, Any]:
        return dict(self.extra)

    def set_extra_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict) or "schema" not in state:
            raise ValueError("extra state must be a dict with a schema")
        self.extra = dict(state)


def test_checkpoint_roundtrips_module_extra_state(tmp_path: Path) -> None:
    source = ModuleWithExtraState()
    source.extra = {"schema": 2, "tags": ["x"], "scale": 0.25}
    assert "_extra_state" in source.state_dict()
    path = tmp_path / "extra.pt"
    checkpoint.save_checkpoint(path, model=source)

    # PyTorch's own safe loader and strict load accept this state.
    reference = ModuleWithExtraState()
    reference.load_state_dict(torch.load(path, weights_only=True)["model"])
    assert reference.extra == source.extra

    # AdaLiFT's validated loader must accept the same checkpoint.
    target = ModuleWithExtraState()
    checkpoint.load_checkpoint(path, model=target)
    assert target.extra == source.extra
    assert_state_equal(target.state_dict(), source.state_dict())

    # A tensor-typed extra state in the file for a module whose extra state is
    # not a tensor is rejected before anything changes.
    state = torch.load(path, weights_only=True)
    state["model"]["_extra_state"] = torch.ones(1)
    torch.save(state, path)
    fresh = ModuleWithExtraState()
    before = copy.deepcopy(fresh.state_dict())
    with pytest.raises(ValueError, match="extra state"):
        checkpoint.load_checkpoint(path, model=fresh)
    assert_state_equal(fresh.state_dict(), before)


class ScalarParameterModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.5))
        self.linear = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.linear(x) * self.scale
        return result


def _trained(module: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    module(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    optimizer.zero_grad()


def test_checkpoint_accepts_scalar_parameters_and_scalar_step_counters(tmp_path: Path) -> None:
    source = ScalarParameterModule()
    source_optimizer = torch.optim.AdamW(source.parameters(), lr=0.01)
    _trained(source, source_optimizer)
    scalar_state = source_optimizer.state[source.scale]
    assert scalar_state["exp_avg"].ndim == 0 and scalar_state["step"].numel() == 1
    path = tmp_path / "scalar.pt"
    checkpoint.save_checkpoint(path, model=source, optimizer=source_optimizer)
    target = ScalarParameterModule()
    optimizer = torch.optim.AdamW(target.parameters(), lr=0.5)
    checkpoint.load_checkpoint(path, model=target, optimizer=optimizer)
    assert_state_equal(optimizer.state_dict(), source_optimizer.state_dict())
    _trained(target, optimizer)  # The restored optimizer can take a step.

    # Python-number step counters (older checkpoints) are accepted too.
    state = torch.load(path, weights_only=True)
    for entry in state["optimizer"]["state"].values():
        entry["step"] = int(entry["step"])
    torch.save(state, path)
    restored = ScalarParameterModule()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.5)
    checkpoint.load_checkpoint(path, model=restored, optimizer=restored_optimizer)
    assert_state_equal(restored.state_dict(), source.state_dict())
    for entry in restored_optimizer.state.values():
        # PyTorch may re-wrap numeric counters as tensors; the value must survive.
        assert float(entry["step"]) == 1.0
    _trained(restored, restored_optimizer)
    assert all(torch.isfinite(parameter).all() for parameter in restored.parameters())


@pytest.mark.parametrize(
    "key,bad_value",
    [
        ("exp_avg", torch.tensor(0.5)),  # scalar moment for a (2, 2) parameter
        ("exp_avg_sq", torch.zeros(4)),  # wrong rank
        ("exp_avg", 0.5),  # not a tensor
        ("step", torch.zeros(2)),  # counter must be a scalar
        ("step", "many"),  # counter must be numeric
        ("step", torch.tensor(True)),  # Boolean tensor crashes the next AdamW step
        ("step", True),  # Boolean Python value
        ("step", torch.tensor(float("nan"))),  # NaN counter corrupts bias correction
        ("step", float("inf")),  # infinite counter
        ("step", torch.tensor(-3.0)),  # negative counter
        ("step", torch.tensor(1 + 0j)),  # complex counter
    ],
)
def test_malformed_optimizer_moments_rejected_before_mutation(
    tmp_path: Path, key: str, bad_value: Any
) -> None:
    source = nn.Linear(2, 2)
    source_optimizer = torch.optim.AdamW(source.parameters(), lr=0.03)
    _trained(source, source_optimizer)
    source_ema = EMA(source, 0.7)
    path = tmp_path / "moments.pt"
    checkpoint.save_checkpoint(
        path,
        model=source,
        optimizer=source_optimizer,
        ema=source_ema,
        capture_rng=True,
        generator=torch.Generator().manual_seed(5),
    )
    state = torch.load(path, weights_only=True)
    weight_index = state["optimizer"]["param_groups"][0]["params"][0]
    state["optimizer"]["state"][weight_index][key] = bad_value
    torch.save(state, path)

    target = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(target.parameters(), lr=0.5)
    _trained(target, optimizer)
    ema = EMA(target, 0.2)
    generator = torch.Generator().manual_seed(9)
    random.seed(99)
    before = copy.deepcopy((target.state_dict(), optimizer.state_dict(), ema.state_dict()))
    torch_before = torch.get_rng_state()
    python_before = random.getstate()
    generator_before = generator.get_state()
    with pytest.raises(ValueError, match="optimizer"):
        checkpoint.load_checkpoint(
            path, model=target, optimizer=optimizer, ema=ema, restore_rng=True, generator=generator
        )
    assert_state_equal((target.state_dict(), optimizer.state_dict(), ema.state_dict()), before)
    assert_state_equal(torch.get_rng_state(), torch_before)
    assert random.getstate() == python_before
    assert_state_equal(generator.get_state(), generator_before)
    _trained(target, optimizer)  # The untouched optimizer still steps.


def test_invalid_extra_state_rejected_by_module_rolls_back(tmp_path: Path) -> None:
    source = ModuleWithExtraState()
    source.extra = {"schema": 3, "tags": [], "scale": 1.0}
    path = tmp_path / "extra_bad.pt"
    checkpoint.save_checkpoint(path, model=source)
    state = torch.load(path, weights_only=True)
    state["model"]["_extra_state"] = {"no_schema": True}  # Dict, but rejected by set_extra_state.
    torch.save(state, path)
    target = ModuleWithExtraState()
    with torch.no_grad():
        target.weight.fill_(-1.0)
    before = copy.deepcopy(target.state_dict())
    extra_before = dict(target.extra)
    with pytest.raises(ValueError, match="schema"):
        checkpoint.load_checkpoint(path, model=target)
    assert_state_equal(target.state_dict(), before)
    assert target.extra == extra_before
