# Review remediation plan

## Source identity

The starting point is the audited local source snapshot, preserved in Git before
changes. The unavailable RC1 ZIP is not reconstructed or claimed as this source.
This work produces RC2 from a new commit and tag.

## Invariants and compatibility

- Adaptive predictions always use an s=1 core pass; max_loops is a hard upper
  bound on per-sample core evaluations, including that pass. min_loops is the
  earliest intermediate check. If there is no room for a check and endpoint,
  complete the fixed schedule. Thus early halting normally uses at least two
  evaluations; max_loops=1 uses one endpoint evaluation.
- Gather only live samples before core execution. loops_used counts executed
  core evaluations; totals equal the sum of executed batch sizes. Intermediate
  residual history uses NaN for inactive samples, rather than stale values.
- Fixed inference reads only its final state. The training trajectory remains
  separately exposed and supervised. Adaptive schedules are an experimental
  extension; endpoint preservation does not establish their quality advantage.
- Checkpoint loading uses the safe PyTorch loader with no unsafe fallback.
  Requested optimizer/EMA state must exist. Versioned checkpoints can optionally
  restore RNG states; caller owns dataset/sampler progress. Legacy safe state
  dictionaries remain loadable.
- Samplers temporarily select evaluation mode and restore each module's mode;
  paired sampling supports identical initial noise. The loss generator covers
  class dropout as well as flow noise, times, and depth coordinates.
- Match the paper's numerical depth schedule and documented training settings;
  do not claim metric reproduction. Keep synthetic execution explicitly a smoke
  runner even when using a paper-scale config.

## Verification before implementation

Tests will instrument core calls using known heterogeneous exit decisions and
check returned endpoint values, sample order, exact counts, hard budgets, and
no-halt equivalence with fixed inference. A discarded endpoint or count mismatch
falsifies the intended adaptive correction. Security tests reject an unapproved
pickle global without executing it. Resume tests compare real optimizer, EMA,
and restored RNG state. Additional regressions cover low precision, invalid
configurations, dropout determinism, distribution contents, and loss gradients.

Run targeted tests, make verify, CPU smoke training, isolated wheel and sdist
tests, and Python 3.11/3.12 checks. Record unavailable GPU checks explicitly.
