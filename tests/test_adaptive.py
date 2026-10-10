import pytest
import torch

from adalift.adaptive import ResidualHaltingPolicy


def test_residual_halting_policy() -> None:
    policy = ResidualHaltingPolicy(threshold=0.2, min_loops=2, max_loops=4)
    current = torch.ones(2, 1, 2, 2)
    previous = current * 0.95
    assert not policy.should_halt(current, previous, 1).any()
    assert policy.should_halt(current, previous, 2).all()


def test_residual_policy_validation() -> None:
    with pytest.raises(ValueError):
        ResidualHaltingPolicy(threshold=-1).validate()
    with pytest.raises(ValueError):
        ResidualHaltingPolicy(min_loops=5, max_loops=4).validate()
    policy = ResidualHaltingPolicy()
    with pytest.raises(ValueError):
        policy.residual(torch.ones(1, 2), torch.ones(1, 3))
