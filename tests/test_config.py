from pathlib import Path

import pytest

from adalift.config import ModelConfig, SamplingConfig, TrainConfig, load_yaml, save_yaml


def test_model_config_validation_errors() -> None:
    with pytest.raises(ValueError):
        ModelConfig(input_size=31, patch_size=2).validate()
    with pytest.raises(ValueError):
        ModelConfig(hidden_size=63, num_heads=4).validate()
    with pytest.raises(ValueError):
        ModelConfig(hidden_size=60, num_heads=6).validate()


def test_more_model_config_validation_errors() -> None:
    with pytest.raises(ValueError):
        ModelConfig(prelude_depth=0).validate()
    with pytest.raises(ValueError):
        ModelConfig(recurrent_depth=0).validate()
    with pytest.raises(ValueError):
        ModelConfig(depth=2, prelude_depth=3).validate()
    with pytest.raises(ValueError):
        ModelConfig(num_classes=0).validate()


def test_train_and_sampling_validation() -> None:
    TrainConfig().validate()
    SamplingConfig().validate()
    with pytest.raises(ValueError):
        TrainConfig(k_train=0).validate()
    with pytest.raises(ValueError):
        TrainConfig(prelude_loss_weight=-1).validate()
    with pytest.raises(ValueError):
        TrainConfig(learning_rate=0).validate()
    with pytest.raises(ValueError):
        TrainConfig(ema_decay=1.0).validate()
    with pytest.raises(ValueError):
        SamplingConfig(num_steps=0).validate()
    with pytest.raises(ValueError):
        SamplingConfig(num_loops=0).validate()


def test_yaml_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    cfg = TrainConfig(k_train=3)
    save_yaml(cfg, path)
    assert load_yaml(path)["k_train"] == 3
    bad = tmp_path / "bad.yaml"
    bad.write_text("- not\n- a\n- mapping\n")
    with pytest.raises(ValueError):
        load_yaml(bad)
