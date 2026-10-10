# Research Log

Use dated entries. Record failed hypotheses as carefully as successful ones.

## 2026-10-08 — Project initialization

- Established clean-room LiFT implementation from the published method.
- Added first AdaLiFT baseline: residual-based per-sample early exit.
- Defined G0-G5 roadmap and first falsification gate.
- No paper reproduction claim yet.

## 2026-10-11 — Independent audit remediation

- Preserved the audited pre-RC1 local source in Git and prepared RC2 from it.
- Fixed endpoint semantics and active-row compute accounting; added behavioral regressions.
- Hardened checkpoint loading, requested-state validation, EMA relocation and opt-in RNG restore.
- Fixed low-precision RoPE and deterministic paired sampling; aligned numerical depth sampling.
- Verified Appendix D: B/2-R2 is published; EMA is 0.999; documented paper settings.
- Added real config smoke execution, distribution checks, pinned CI dependencies and Actions.
- Evidence remains synthetic implementation verification. No metric reproduction or adaptive advantage.
- Hosted CI caught an inherited Git ignore error: `data/` omitted the package's
  synthetic-data source from the initial commit. Anchored `/data/`, restored both
  source files to Git, and checked the release from a fresh Git snapshot.
