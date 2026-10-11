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
