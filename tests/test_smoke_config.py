import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from adalift.training.smoke import _git_identity, main, read_run_config


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


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.com",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.com",
        },
    ).stdout.strip()


@pytest.mark.skipif(shutil.which("git") is None, reason="requires git")
def test_git_identity_ignores_unrelated_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A separate repository containing a copy of the package.
    package_repo = tmp_path / "pkg_repo"
    package_dir = package_repo / "src" / "adalift" / "training"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("")
    _git("init", "-q", cwd=package_repo)
    _git("add", "-A", cwd=package_repo)
    _git("commit", "-q", "-m", "package", cwd=package_repo)
    package_commit = _git("rev-parse", "HEAD", cwd=package_repo)

    # An unrelated repository that the caller happens to run from.
    other_repo = tmp_path / "other_repo"
    other_repo.mkdir()
    _git("init", "-q", cwd=other_repo)
    _git("commit", "-q", "--allow-empty", "-m", "unrelated", cwd=other_repo)
    other_commit = _git("rev-parse", "HEAD", cwd=other_repo)
    assert other_commit != package_commit
    monkeypatch.chdir(other_repo)

    identity = _git_identity(package_dir)
    assert identity == {"commit": package_commit, "dirty": False}
    (package_dir / "__init__.py").write_text("# changed\n")
    assert _git_identity(package_dir)["dirty"] is True

    # A package directory inside a repository that does not track it reports null.
    untracked = other_repo / "site-packages" / "adalift" / "training"
    untracked.mkdir(parents=True)
    (untracked / "__init__.py").write_text("")
    assert _git_identity(untracked) == {"commit": None, "dirty": None}

    # No repository at all reports null.
    plain = tmp_path / "plain" / "adalift" / "training"
    plain.mkdir(parents=True)
    (plain / "__init__.py").write_text("")
    assert _git_identity(plain) == {"commit": None, "dirty": None}
