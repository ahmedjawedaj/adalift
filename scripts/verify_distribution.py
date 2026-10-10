"""Build and test a wheel and sdist in disposable, non-editable environments."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import venv
from pathlib import Path


def run(args: list[str], cwd: Path) -> None:
    subprocess.run(
        args,
        cwd=cwd,
        check=True,
        env={
            **os.environ,
            "PYTHONPATH": "",
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_INDEX_URL": "https://pypi.org/simple",
            "PIP_EXTRA_INDEX_URL": "",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, help="copy verified wheel/sdist here on success")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if (root / ".git").exists():
        tracked = set(
            subprocess.run(
                ["git", "ls-files", "src/adalift"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
        )
        source_files = {
            str(path.relative_to(root))
            for path in (root / "src/adalift").rglob("*")
            if path.is_file() and (path.suffix == ".py" or path.name == "py.typed")
        }
        missing = sorted(source_files - tracked)
        if missing:
            raise RuntimeError(f"package source missing from Git: {missing}")
    with tempfile.TemporaryDirectory(prefix="adalift-dist-") as temporary:
        workspace = Path(temporary)
        dist = workspace / "dist"
        run([sys.executable, "-m", "build", "--outdir", str(dist)], root)
        run([sys.executable, "-m", "twine", "check", *map(str, dist.iterdir())], root)
        wheel = next(dist.glob("*.whl"))
        sdist = next(dist.glob("*.tar.gz"))
        extracted = workspace / "sdist"
        with tarfile.open(sdist) as archive:
            archive.extractall(extracted, filter="data")
        source = next(extracted.iterdir())
        for required in ("docs", "configs", "scripts", ".agents", ".claude", "CITATION.cff"):
            if not (source / required).exists():
                raise RuntimeError(f"sdist missing {required}")
        for kind, artifact, test_source in (("wheel", wheel, root), ("sdist", source, source)):
            environment = workspace / f"{kind}-venv"
            uv = shutil.which("uv")
            if uv is not None:
                run([uv, "venv", "--python", sys.executable, "--seed", str(environment)], workspace)
            else:
                venv.EnvBuilder(with_pip=True).create(environment)
            python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            staging = workspace / f"{kind}-tests"
            staging.mkdir()
            for directory in ("tests", ".agents", ".claude", "configs"):
                shutil.copytree(test_source / directory, staging / directory)
            shutil.copy2(test_source / "pyproject.toml", staging / "pyproject.toml")
            requirements = [str(artifact) + "[dev]"]
            if sys.platform == "linux":
                run(
                    [
                        str(python),
                        "-m",
                        "pip",
                        "install",
                        "--index-url",
                        "https://download.pytorch.org/whl/cpu",
                        "torch==2.14.1",
                    ],
                    staging,
                )
            command = [str(python), "-m", "pip", "install", *requirements]
            constraints = root / "requirements-ci.txt"
            if constraints.exists():
                command.extend(["-c", str(constraints)])
            run(command, staging)
            run(
                [
                    str(python),
                    "-c",
                    "import adalift; print('installed package:', adalift.__file__)",
                ],
                staging,
            )
            run(
                [str(python), "-m", "pytest", "--cov=adalift", "--cov-report=term-missing"], staging
            )
            executable = environment / (
                "Scripts/adalift-smoke.exe" if os.name == "nt" else "bin/adalift-smoke"
            )
            run(
                [
                    str(executable),
                    "--steps",
                    "2",
                    "--batch-size",
                    "2",
                    "--config",
                    str(staging / "configs/mini_lift.yaml"),
                ],
                staging,
            )
        if args.artifacts is not None:
            args.artifacts.mkdir(parents=True, exist_ok=True)
            for artifact in (wheel, sdist):
                shutil.copy2(artifact, args.artifacts / artifact.name)
        print("wheel and sdist passed isolated installation, tests, and config smoke runs")


if __name__ == "__main__":
    main()
