---
name: experiment-review
description: Review an AdaLiFT experiment before treating it as evidence or adding it to a research claim.
---

Use before accepting experimental results.

1. Confirm run provenance: commit, config, seed, dataset, software/hardware, precision, sample count.
2. Check whether comparison budgets are actually matched.
3. Look for leakage from evaluation choices, changed samplers, unequal EMA, or different preprocessing.
4. Require multiple seeds for extension claims unless the task is explicitly exploratory.
5. Compare wall-clock/VRAM as well as analytic FLOPs when making efficiency claims.
6. Separate observed metric change from causal interpretation.
7. Record plausible alternative explanations and the next falsification test.
