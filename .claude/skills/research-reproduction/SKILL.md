---
name: research-reproduction
description: Apply AdaLiFT's evidence standard when reproducing LiFT or comparing against paper results.
---

Use for any paper reproduction, benchmark, or claim-comparison task.

Read `docs/REPRODUCIBILITY.md` and `docs/PAPER_BREAKDOWN.md` first.

Required workflow:
1. Identify the exact paper claim being tested.
2. Record architecture, data, optimizer, training budget, seed, sampler, K_train/K_inf, and metric protocol.
3. Mark unverified paper details explicitly; never guess them into a reproduction claim.
4. Match baselines on the dimension stated by the experiment: parameters, executed depth, FLOPs, or training tokens.
5. Use common noise/labels where paired comparisons benefit from lower variance.
6. Report negative results and implementation deviations.
7. Classify evidence as smoke verified, behavior reproduced, metric reproduced, or extension validated.
