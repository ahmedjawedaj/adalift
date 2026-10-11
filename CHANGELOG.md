# Changelog

## 0.1.0rc4 — 2026-10-12

- First public research-alpha release, including the independently reviewed
  checkpoint follow-up: safe module extra state, strict moment shapes, finite
  real non-negative optimizer counters, and transactional rollback regressions.
- New README diagrams, runnable adaptive-depth example, FAQ, contribution
  workflow, maintainer guide, and repository discovery metadata.
- Canonical Apache-2.0 license text, named governance/contact, code-owner
  routing, issue/PR guidance, and publication protection payloads.
- Dependency audits on PRs and main pushes; pinned audit tooling and checkout
  credentials disabled. Owner/admin review bypass is separate from mandatory CI.
- No reproduced ImageNet/FID metric, pretrained weights, production stability,
  or adaptive quality/latency advantage is claimed.

## 0.1.0rc3 — private review candidate

- Smoke run records identify the Git checkout containing the installed package
  rather than the caller's working directory, and report null when the package
  is not tracked by a repository.
- `load_checkpoint` validates model, optimizer, EMA and RNG compatibility before
  mutating any object and rolls back on failure, so a failed load leaves the
  caller's model, optimizer, EMA and requested RNG states unchanged.
  Return metadata is validated before application. `EMA.validate_state_dict`
  is public. Custom load hooks must avoid irreversible external side effects.
  Factored optimizer statistics remain loadable. Module extra state
  (`get_extra_state`/`set_extra_state`) round-trips through the safe loader.
  Known optimizer moments must match their parameter shape, scalars included,
  and step counters must be finite, real, non-negative scalars.
- Tests cover asymmetric EMA weighting, generator-driven class dropout,
  residual normalization, sampler callback isolation, all-or-nothing checkpoint
  loading and Git identity resolution, including malformed RNG/metadata and
  an injected late failure that exercises rollback.
- README and paper breakdown name the exact B/2-R2 fields the candidate config
  encodes and distinguish the 500k main protocol from the 100k ablations.

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

## Initial bootstrap (before release candidates)

### Added
- Clean-room LiFT architecture and trajectory objective.
- Fixed and residual-adaptive recurrent inference.
- Euler flow sampler.
- EMA/checkpoint utilities.
- Synthetic-latent smoke training.
- Tests, CI, contribution workflow, and agent skills.
