import json
import sys
from pathlib import Path

import pytest

from adalift.training.smoke import main, read_run_config


def test_config_runner_reads_real_configs() -> None:
    root = Path(__file__).resolve().parents[1]
    model, train, _ = read_run_config(root / "configs/mini_lift.yaml")
    assert model.hidden_size == 128 and train.k_train == 2
    model, train, _ = read_run_config(root / "configs/lift_b2_r2_candidate.yaml")
    assert (model.hidden_size, model.num_heads, model.prelude_depth, model.recurrent_depth) == (
        768,
        12,
        4,
        2,
    )
    assert train.k_train == 4 and train.ema_decay == 0.999


def test_config_runner_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("model:\n  misspelled_width: 12\n")
    with pytest.raises(TypeError):
        read_run_config(path)
    path.write_text("unknown_section: {}\n")
    with pytest.raises(ValueError):
        read_run_config(path)


def test_smoke_writes_configured_run_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run.json"
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "adalift-smoke",
            "--steps",
            "2",
            "--batch-size",
            "2",
            "--config",
            str(root / "configs/mini_lift.yaml"),
            "--output",
            str(output),
        ],
    )
    main()
    record = json.loads(output.read_text())
    assert record["evidence_level"] == "smoke verified"
    assert record["config"]["model"]["hidden_size"] == 128
    assert record["dataset"]["type"] == "synthetic"
    assert record["steps"] == len(record["metrics"]) == 2
    assert record["environment"]["precision"] == "float32"


def test_default_smoke_config_and_bad_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    model, train, _ = read_run_config(None)
    assert model.input_size == 16 and train.warmup_steps == 0
    monkeypatch.setattr(sys, "argv", ["adalift-smoke", "--steps", "0"])
    with pytest.raises(SystemExit):
        main()
