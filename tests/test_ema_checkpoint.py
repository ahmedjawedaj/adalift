from pathlib import Path

import torch
from torch import nn

from adalift.training.checkpoint import load_checkpoint, save_checkpoint
from adalift.training.ema import EMA


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema = EMA(model, 0.9)
    path = tmp_path / "state.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, ema=ema, step=17, metadata={"x": 1})
    with torch.no_grad():
        model.weight.zero_()
    state = load_checkpoint(path, model=model, optimizer=optimizer, ema=ema)
    assert state["step"] == 17
    assert state["metadata"] == {"x": 1}
    assert not torch.equal(model.weight, torch.zeros_like(model.weight))
