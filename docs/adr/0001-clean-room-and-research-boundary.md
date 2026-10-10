# ADR-0001: Clean-room implementation and research boundary

**Status:** Accepted
**Date:** 2026-10-08

## Decision

AdaLiFT implements the published LiFT method from paper-level descriptions without copying unreleased author code. Paper-faithful behavior lives in the LiFT model/objective path; experimental adaptive-depth mechanisms are named and documented as AdaLiFT extensions.

## Why

This keeps licensing/provenance clear and prevents extension behavior from being mistaken for original-paper behavior. It also makes future comparison with official code explicit and testable.

## Consequences

- Official code, when released, is a compatibility/reference source rather than a silent replacement.
- Deviations discovered later are documented in `docs/PAPER_BREAKDOWN.md` and the changelog.
- New research mechanisms must not be hidden behind defaults that alter paper-faithful behavior.
