# Experiment Matrix

Every experiment should have a hypothesis, matched baseline, success criterion, and stop condition.

| ID | Question | Baseline | Treatment | Primary metric |
|---|---|---|---|---|
| E01 | Does useful depth extrapolate? | `K_inf=K_train` | larger `K_inf` | FID / task metric vs FLOPs |
| E02 | Does trajectory supervision matter? | endpoint target at every loop | LiFT path target | quality at `K_inf>K_train` |
| E03 | Do random depth coordinates matter? | fixed training grid | random sorted grid | extrapolation stability |
| E04 | Does reinjection matter? | no `h0` reinjection | LiFT boundary | quality/stability |
| E05 | Does depth conditioning matter? | no `e_s` | continuous `e_s(s)` | extrapolation |
| E06 | Is residual halting useful? | fixed K | residual adaptive K | quality at matched FLOPs |
| E07 | Can halting be learned? | E06 | learned halting | Pareto frontier |
| E08 | How to allocate compute? | fixed T/K | adaptive T/K or K(t) | quality/latency |

Do not add new rows to result tables without a committed config and machine-readable run record.
