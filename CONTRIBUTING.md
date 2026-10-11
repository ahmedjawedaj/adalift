# Contributing

Thanks for helping build AdaLiFT. This is a research codebase, so correctness includes both software behavior and scientific claims.

## Find a starting point

You can contribute without training a large model:

- Run the [forward](examples/minimal_forward.py) and [adaptive-depth](examples/adaptive_depth.py) CPU examples. Improve an explanation that was hard to follow or report a reproducible error.
- Review a paper-to-code mapping in [PAPER_BREAKDOWN.md](docs/PAPER_BREAKDOWN.md), citing the paper section and exact implementation lines.
- Add a focused regression test for a demonstrated bug, including the failing input and expected behavior.
- Propose an experiment from the [roadmap](docs/ROADMAP.md), with a hypothesis, baseline, quality metric, and compute protocol.

Browse [existing issues](https://github.com/ahmedjawedaj/adalift/issues) before [opening an issue](https://github.com/ahmedjawedaj/adalift/issues/new/choose). Use the bug, feature, or research proposal template that fits your contribution. If you find a security issue, follow [SECURITY.md](SECURITY.md).

## Before opening a PR

1. Open or reference an issue for non-trivial changes.
2. For model/objective changes, write the hypothesis and expected failure mode before coding.
3. For behavior changes, add or update tests that demonstrate the failure and the fix. For documentation changes, run any new examples and check links.
4. Run `make verify`.
5. For experiment changes, attach the exact config, seed, commit, hardware, and metric protocol.
6. Do not report a paper result as reproduced unless the reproduction checklist in `docs/REPRODUCIBILITY.md` is satisfied.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements-ci.txt -e ".[dev]"
pre-commit install
make verify
```

The verified development matrix is Python 3.11 and 3.12. On Windows, activate with `.venv\Scripts\Activate.ps1`. If `make` is unavailable, run the same checks individually:

```bash
python scripts/sync_agent_skills.py --check
ruff check .
ruff format --check .
mypy src/adalift
pytest --cov=adalift --cov-report=term-missing
```

## Pull request scope

Keep PRs reviewable. Prefer one scientific or engineering change per PR. Large experiments should separate reusable infrastructure from result artifacts.

Describe the problem, the resulting behavior, and how you verified it. Link the issue and call out scientific assumptions or incomplete validation. Include the config and machine-readable results for experiments. A maintainer can then review the evidence and request focused follow-ups.

## Fork and pull request workflow

Fork the repository on GitHub. Replace `YOUR_USERNAME` below with your own GitHub account:

```bash
git clone https://github.com/YOUR_USERNAME/adalift.git
cd adalift
git remote add upstream https://github.com/ahmedjawedaj/adalift.git
git fetch upstream
git switch -c fix/short-description upstream/main
```

Set up the development environment above, make one focused change, and run the relevant checks. Commit the intended files, push your branch to your fork, and open a pull request against `ahmedjawedaj/adalift:main`. Draft PRs are welcome for early design feedback; clearly state what remains unfinished.

The publication policy requires passing CI, a current branch, one approving review from a trusted reviewer other than the author for external contributions, and resolved review conversations. The owner/admin may bypass the approval requirement for their own PRs after self-review; the separate CI requirement still applies. New changes dismiss old approvals. Maintainers squash-merge accepted PRs; finished branches are deleted automatically. See the [maintainer guide](docs/MAINTAINER_GUIDE.md) for the exact protection configuration and its activation status.

Fork CI has no repository secrets or write token. Public fork workflow runs require approval under the publication policy described in the maintainer guide; verify the live setting when maintaining the repository. A maintainer should inspect workflow and dependency changes before approving a run. External reviewers can comment without write access, but their comments alone do not satisfy GitHub's required approving review.

## License and attribution

Submit work you have the right to contribute. Contributions are made under the project's [Apache-2.0 license](LICENSE); retain applicable third-party license and attribution notices. Reference the source of adapted algorithms or scientific claims. Do not include datasets, checkpoints, secrets, or material whose redistribution rights are unclear.

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
