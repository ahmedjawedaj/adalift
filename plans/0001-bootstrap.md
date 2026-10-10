# ExecPlan 0001 — Bootstrap AdaLiFT

## Goal

Create a public-contribution-ready research repository with a clean-room LiFT core, deterministic verification, contributor workflows, and an initial adaptive-depth baseline.

## Non-goals

No ImageNet headline reproduction claim, no pretrained checkpoint, and no claim that residual halting improves quality yet.

## Scientific hypothesis

Small-model depth extrapolation is a diagnostic. Assess sufficiently capable
paper-scale configurations before rejecting the hypothesis. Endpoint-preserving
adaptive inference remains experimental until matched-compute validation.

## Implementation

- LiFT prelude/core/head architecture
- continuous depth conditioning and boundary reinjection
- flow trajectory objective and prelude auxiliary loss
- fixed and residual-adaptive Euler sampling
- dense baseline using identical block design
- deterministic synthetic smoke path
- EMA/checkpoint utilities
- 85%+ test-coverage gate
- CI, OSS governance, issue/PR templates
- Codex/Claude project skills

## Verification

- CPU unit tests
- coverage >= 85%
- editable package build/install
- wheel build
- smoke training
- git whitespace check
- skill synchronization check

## Progress

- 2026-10-08: Core implementation, dense baseline, adaptive residual baseline, tests, docs, CI, and skills completed.

## Outcome

Foundation is ready for G1 mini-LiFT experiment work. GPU-scale paper reproduction remains intentionally unclaimed.
