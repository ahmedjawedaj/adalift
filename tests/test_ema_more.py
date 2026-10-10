import pytest
import torch
from torch import nn

from adalift.training.ema import EMA


def test_ema_update_and_context_restore() -> None:
    model = nn.Linear(2, 1, bias=False)
    ema = EMA(model, decay=0.5)
    original_shadow = ema.shadow["weight"].clone()
    with torch.no_grad():
        model.weight.add_(2.0)
    live = model.weight.detach().clone()
    ema.update(model)
    expected = original_shadow * 0.5 + live * 0.5
    torch.testing.assert_close(ema.shadow["weight"], expected)
    with ema.average_parameters(model):
        torch.testing.assert_close(model.weight, expected)
    torch.testing.assert_close(model.weight, live)


def test_ema_state_roundtrip_and_validation() -> None:
    model = nn.Linear(2, 1)
    ema = EMA(model, 0.8)
    state = ema.state_dict()
    other = EMA(model, 0.5)
    other.load_state_dict(state)
    assert other.decay == 0.8
    with pytest.raises(ValueError):
        EMA(model, 1.0)
    with pytest.raises(TypeError):
        other.load_state_dict({"decay": 0.9, "shadow": []})
