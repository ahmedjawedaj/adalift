# RC2 independent review handoff

## Artifact and scope

Repository: https://github.com/ahmedjawedaj/adalift (private during review).
Version: 0.1.0rc2. Initial Git baseline:
`f80ca7328b8f28bd177142f5d71728a841bd22c4`.
RC2 is a remediation of that snapshot, not a recovered or verified RC1 ZIP.
The audited working tree had 83 files. The first Git baseline contained only
81: its inherited unanchored `data/` ignore rule omitted the two files in
`src/adalift/data`. Hosted CI exposed the omission. RC2 anchors `/data/` to the
repository root and includes both package files. The original audit manifest
in the review bundle preserves the full starting working-tree identity.
Review the candidate tag, archive hash and diff from the starting commit.
The sibling `adalift-review-rc2` bundle records final commit/hash and check logs.

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

Local Python 3.11 and 3.12 checks pass: Ruff, format, Mypy, synchronized skills,
66 CPU tests and one explicitly skipped CUDA test, with 89.78% coverage. The
final bundle additionally records minimum-version and distribution checks and
the hosted CI result. Commands:

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
