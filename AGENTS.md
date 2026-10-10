# AdaLiFT agent instructions

## Project overview

AdaLiFT is a research-grade, clean-room PyTorch implementation of LiFT plus adaptive recurrent-depth extensions. Core code is under `src/adalift/`; tests under `tests/`; research protocols under `docs/`.

## Mandatory workflow triggers

- Before changing model architecture, losses, samplers, experiment semantics, or public APIs, use `$implementation-strategy`.
- For paper reproduction or benchmark claims, use `$research-reproduction`.
- After changing runtime code, tests, configs, CI, or packaging, use `$code-change-verification`.
- Before accepting an experiment result as evidence, use `$experiment-review`.
- When substantial work is ready for review, use `$pr-draft-summary`.

Repo-local Codex skills live in `.agents/skills/`.

## Engineering rules

- Keep core dependencies small; optional research integrations belong in extras.
- Preserve deterministic CPU tests for every core path.
- Do not silently change research semantics to make a test pass.
- Do not call a result reproduced unless `docs/REPRODUCIBILITY.md` requirements are satisfied.
- Keep model logic separate from data/evaluation/training orchestration.
- Public APIs need type hints. Prefer explicit tensors/configs over hidden globals.

## Verification

Run `make verify` for substantial changes. Local tests use disposable synthetic fixtures and have no production access; fix failures caused by your change and rerun affected tests without asking for approval at each step.
