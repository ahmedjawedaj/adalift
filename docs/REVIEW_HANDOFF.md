# RC3 independent review handoff

Historical review record for the RC3 source and follow-up. The first public
release is RC4; see the changelog and GitHub release evidence for its identity.

## Artifact and scope

Repository: https://github.com/ahmedjawedaj/adalift (private during review).
Version: 0.1.0rc3. Initial Git baseline:
`f80ca7328b8f28bd177142f5d71728a841bd22c4`.
RC2 is a remediation of that snapshot, not a recovered or verified RC1 ZIP.
The audited working tree had 83 files. The first Git baseline contained only
81: its inherited unanchored `data/` ignore rule omitted the two files in
`src/adalift/data`. Hosted CI exposed the omission. RC2 anchors `/data/` to the
repository root and includes both package files. The original audit manifest
in the review bundle preserves the full starting working-tree identity.
Review the candidate tag, archive hash and diff from the starting commit.
The original `adalift-review-rc2` bundle is preserved. The sibling
`adalift-review-rc3` bundle records the final RC3 commit/hash and check logs.

## RC3 follow-up

The supplied RC2 review found no critical, high or medium defects. It exercised
independent adaptive, checkpoint, precision and distribution probes. Its model
identity was configured as Sonnet 5.5; actual serving identity was not verified.
The external fix addendum discloses a serving-model change to Fable 5.1 during
the cleanup. This repository does not label those fixes as a Sonnet review.

RC3 records the containing package's tracked Git source rather than the caller's
cwd. Checkpoint prevalidation now covers model, known optimizer statistics,
EMA, return metadata and requested RNG compatibility; failed application rolls
back model, optimizer, EMA and requested RNG states. New tests cover the four
mutation gaps and actually exercise a late failure after RNG restoration.
Custom hooks with irreversible external effects remain outside this guarantee.
Valid factored optimizer statistics are accepted. `ModelConfig.depth` is the
DenseDiT block count; LiFT executes prelude_depth + K * recurrent_depth.

Candidate wording covers architecture and optimizer fields, with the 500,000
main updates distinguished from the 100,000-update ablations. No ImageNet or
BF16 training pipeline is implied by the synthetic FP32 smoke runner.

## Changes mapped to the independent audit

| Finding | Remediation / reviewer focus |
|---|---|
| C-1 identity | Git baseline plus original 83-file audit manifest; corrected ignored package files; candidate tagged and ZIP generated from Git, with hashes. |
| H-1 endpoint | Early exits run a final core pass at s=1. Budget exhaustion already ends at s=1. |
| H-2 compute | Gather/scatter live rows; counts include endpoint calls and match executed row counts. |
| H-3 checkpoint | weights_only=True; no fallback; plain metadata; rejected-global regression. |
| H-4 CI | Ruff, format, Mypy, coverage gates; pinned tooling/constraints and Actions. |
| M-1 distributions | Source assets included; wheel and sdist tested in separate clean installations. |
| M-2 metadata | Real private repository URLs and security contact, central package version. |
| M-3 precision | RoPE returns input dtype; pure bf16/fp16 regression tests for both models. |
| M-4 resume | Requested optimizer/EMA state required; EMA moves to model dtype/device; opt-in RNG restore. |
| M-5 determinism | Generator includes class dropout; samplers select eval mode and restore caller modes. |
| M-6 assertions | Controlled heterogeneous exits, endpoint values, exact accounting, budget limits, ordering, loss formula, actual state restoration. |
| M-7 configs | YAML-backed synthetic CLI and run record. Paper-scale config remains a candidate. |
| L-1 to L-9 | Shared initial noise, residual NaNs for inactive rows, validation, final-only default head, typed wheel, smaller runtime dependencies, reproducible CI constraints. |
| L-10 | Skill sync preserves extra contributor skills instead of deleting the destination tree. |
| L-11 | Full LiFT author list and research acknowledgements; no false release date. |
| L-12 | Locked dependencies audited; minimum raised to Torch 2.13, pytest upgraded to 9.1.1. |

## Compatibility and scientific decisions

`loops_used` now means actual per-sample recurrent-core evaluations, including
the endpoint. `max_loops` is a hard cap on that count; `min_loops` is the first
intermediate residual check. Early exit at check k normally costs k+1; budget 1
is an endpoint-only pass. Counts exclude head/prelude/compaction overhead and
must not be presented as FLOPs or wall-clock savings.

Residuals compare predictions at different depth coordinates. An endpoint pass
restores the endpoint task, but does not certify convergence or quality. The
adaptive path is explicitly experimental. Inactive entries in residual history
are NaN, and endpoint passes do not add residual entries.

Sampler calls temporarily select evaluation mode and restore each module's
previous mode, including on exceptions. Direct model calls preserve caller mode.
Checkpoint legacy plain state dictionaries remain accepted; requested missing
states now raise. Metadata is restricted to plain values. RNG restoration is
opt-in and does not restore external/data-loader progress automatically.

The training objective remains LiFT. Numerical depth draws/sort now follow
Appendix D.3 (FP64 draws, epsilon spacing, FP32 coordinates). Training defaults
record the paper's EMA 0.999 and optimizer settings. B/2-R2 at K_train=4 is in
Table 15; its previous absence from Table 1 was not an architecture defect.

## Verification and remaining limits

The initial independently reviewed RC2 had 66 passing CPU tests, one skipped CUDA test
and 89.78% coverage. Local RC3 checks pass on Python 3.11/3.12 and minimum
PyTorch 2.13.0 at the initial RC3 commit: 80 passing CPU tests, one skipped CUDA test and 89.26% coverage,
plus Ruff, formatting, Mypy and skill sync. Its final bundle records the
distribution and hosted CI results for that source commit.

The subsequent 2026-10-12 working-tree review independently reproduces the
checkpoint follow-up: 12 failures on `a4a4cf3`, all 22 checkpoint transaction
tests passing on the patched tree. The complete current suite has 94 passing
CPU tests, one skipped CUDA test and 89.88% coverage on Python 3.11/3.12 with
locked PyTorch, and Python 3.11 with minimum PyTorch 2.13.0. Wheel/sdist clean
installation, full suites and config smoke runs also pass. The working-tree
changes remain uncommitted; hosted CI for `a4a4cf3` does not verify them.
Public contribution and protection status is recorded in
[PUBLIC_RELEASE_REVIEW.md](PUBLIC_RELEASE_REVIEW.md).

Commands:

```bash
python -m pip install -c requirements-ci.txt -e ".[dev]"
make verify
adalift-smoke --config configs/mini_lift.yaml --steps 3 --batch-size 2
python scripts/verify_distribution.py
```

The suite emits PyTorch's optional NumPy initialization warning when NumPy is
absent; core execution does not require NumPy. GPU/CUDA, MPS, distributed and
large-scale behavior remain unverified. The smoke CLI is FP32 and synthetic,
including with a B/2 config. ImageNet preprocessing, evaluator/metric reproduction
and a multi-seed matched-compute adaptive study remain future work.

Do not promote this candidate to public release before independent review and
checking hosted CI. No reproduced paper metric or adaptive advantage is claimed.
