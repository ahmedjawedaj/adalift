# Changelog

## 0.1.0rc2 — 2026-10-11

- Endpoint-preserving adaptive inference with active-row compaction and honest
  core-evaluation accounting, bounded by max_loops.
- Safe checkpoint loading, strict requested-state validation, EMA relocation,
  and optional RNG capture/restoration.
- Low-precision RoPE, seeded class dropout, common-noise sampling, mode restoration,
  FP64 depth sampling with paper spacing, and input/config validation.
- Verified B/2-R2 candidate settings, real YAML smoke execution, run metadata,
  central versioning, sdist assets, typed package and console entry point.
- Pinned CI dependencies and Actions, distribution validation, updated citation,
  real repository metadata, and expanded behavioral regressions.
- No reproduced paper metric or adaptive quality/latency advantage is claimed.

All notable changes are documented here.

## [Unreleased]

### Added
- Clean-room LiFT architecture and trajectory objective.
- Fixed and residual-adaptive recurrent inference.
- Euler flow sampler.
- EMA/checkpoint utilities.
- Synthetic-latent smoke training.
- Tests, CI, contribution workflow, and agent skills.
