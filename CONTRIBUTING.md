# Contributing

Thanks for helping build AdaLiFT. This is a research codebase, so correctness includes both software behavior and scientific claims.

## Before opening a PR

1. Open or reference an issue for non-trivial changes.
2. For model/objective changes, write the hypothesis and expected failure mode before coding.
3. Add or update tests.
4. Run `make verify`.
5. For experiment changes, attach the exact config, seed, commit, hardware, and metric protocol.
6. Do not report a paper result as reproduced unless the reproduction checklist in `docs/REPRODUCIBILITY.md` is satisfied.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
make verify
```

## Pull request scope

Keep PRs reviewable. Prefer one scientific or engineering change per PR. Large experiments should separate reusable infrastructure from result artifacts.

## Code expectations

- Python 3.11+.
- Public APIs need type hints and concise docstrings.
- No hidden global state for seeds, devices, or experiment configuration.
- Tests should be deterministic and CPU-capable unless explicitly marked `gpu`.
- Avoid optional heavyweight dependencies in the core package.
- New research claims require a falsification test, not just a best-case metric.

## Research artifacts

Do not commit large checkpoints or datasets. Store small machine-readable metric summaries under `experiments/` and link immutable external artifacts from the experiment card.

## Agent-assisted contributions

Agent-generated code is welcome, but the contributor remains responsible for correctness. `AGENTS.md`, `CLAUDE.md`, and repo-local skills define the expected workflows.
