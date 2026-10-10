---
name: implementation-strategy
description: Plan model, loss, sampler, public API, or cross-module changes before editing AdaLiFT runtime code.
---

Use this skill before changing architecture, training semantics, samplers, checkpoint formats, or public APIs.

1. Read only the relevant architecture/reproducibility docs and touched modules.
2. State the invariant being changed and the compatibility boundary.
3. Separate paper-faithful behavior from AdaLiFT extensions.
4. Identify the smallest implementation that preserves module boundaries.
5. Define tests before implementation, including at least one failure/edge case.
6. For research semantics, define what result would falsify the change's intended effect.
7. Do not broaden scope unless required for correctness.
