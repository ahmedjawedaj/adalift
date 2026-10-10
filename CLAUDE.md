# AdaLiFT project context

AdaLiFT is a research-grade PyTorch implementation of LiFT plus adaptive recurrent-depth extensions.

## Where things live

- Core package: `src/adalift/`
- Tests: `tests/`
- Research protocol: `docs/REPRODUCIBILITY.md`
- Roadmap: `docs/ROADMAP.md`
- Project skills: `.claude/skills/`

## Use project skills

Use the relevant project skill before architecture/loss/sampler changes, reproduction claims, experiment interpretation, verification, or PR handoff. Skills mirror the canonical `.agents/skills/` definitions; keep them synchronized with `make sync-skills`.

## Core rules

- This is a clean-room implementation. Do not copy unreleased or incompatible-licensed code.
- Never trade scientific correctness for a passing test.
- Distinguish smoke verification, behavior reproduction, metric reproduction, and extension validation.
- Core code must retain deterministic CPU coverage.
- Keep heavyweight metrics/loggers optional.
- Run `make verify` after substantial changes.
