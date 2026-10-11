# RC3 review follow-up

RC3 incorporates the externally supplied eleven-file cleanup of RC2. Preserve
the reviewed RC2 tag and bundle; commit the final candidate separately.

## Invariants and compatibility

- Smoke provenance identifies tracked package source, never the caller's cwd.
- Validate checkpoint structure, return metadata and requested training/RNG
  states before applying them. If application fails, restore model, optimizer,
  EMA and requested RNG states. Custom load hooks with irreversible external
  effects cannot be made transactional by this library.
- Keep safe loading, legacy format compatibility and opt-in RNG restoration.
- Optimizer state may contain scalar counters and factored statistics: do not
  reject valid optimizer-specific shapes through a generic parameter-shape rule.
- No changes to LiFT training or adaptive inference semantics. Narrow candidate
  configuration wording without claiming a full reproduction pipeline.

## Verification

Retain the new asymmetric EMA, dropout-generator, residual and callback tests.
Add malformed RNG/return-metadata cases and inject a late failure after RNG
application to exercise rollback, asserting actual state equality. A changed
model, optimizer, EMA or RNG after a failed application falsifies the intended
transaction. Check valid factored optimizer state still loads.

Run targeted tests, make verify on Python 3.11/3.12 and minimum PyTorch,
isolated wheel/sdist checks, then hosted CI on the committed candidate. Tag and
archive only the passing commit. GPU and research metrics remain unverified.
