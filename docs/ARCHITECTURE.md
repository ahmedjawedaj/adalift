# Architecture

## Scope

AdaLiFT is a clean-room implementation of the published LiFT method plus adaptive-depth extensions. The core package operates on latent tensors; VAE/tokenizer integration is deliberately kept outside the model core so alternative latent tokenizers can be tested without rewriting LiFT.

## Model path

```text
x_t
 │
 ├─ patch embedding
 │
 ├─ prelude blocks Pθ ────────────────┐
 │                                    │ h0
 │                                    ▼
 │                               prediction head ── u0 (training anchor)
 │                                    │
 │ h = h0                             │
 │   │                                │
 │   ├─ RMSNorm(h) + h0               │
 │   ├─ shared core Rθ, conditioned on c + e_s(s1)
 │   ├─ prediction head ────────────── u1
 │   ├─ RMSNorm(h) + h0
 │   ├─ shared core Rθ, conditioned on c + e_s(s2)
 │   └─ ... ────────────────────────── uK
 │
 └─ outer Euler/ODE sampler advances x_t using only uK at inference
```

The outer condition is `c = e_t(t) + e_y(y)`. The depth embedding is supplied only to the recurrent core. The prediction head is shared across depths and does not alter recurrent state.

## Training objective

For data latent `x1`, Gaussian noise `x0`, and `t ~ U(0,1)`:

```text
x_t = (1 - t) x0 + t x1
u*  = x1 - x0
b   = stop_gradient(u0)
ubar(s) = (1 - s) b + s u*
```

For `K_train > 1`, draw and sort `K_train - 1` interior coordinates in FP64.
Apply the paper's spacing rule `s_i = (1 - K * 1e-6) * r_i + i * 1e-6`,
then cast coordinates to FP32 and append the endpoint `1`. Training depth
coordinates remain FP32 even when the model uses lower precision. Loss
accumulation is FP32. Each readout is supervised against `ubar(s_k)`.

The total loss is the average recurrent trajectory MSE plus a small auxiliary prelude MSE weighted by `lambda`.

## Adaptive-depth extension

`LiFT.forward_adaptive` is an experimental AdaLiFT extension. Each normal pass
uses `s=k/max_loops`. `ResidualHaltingPolicy` compares consecutive readouts.
When its criterion fires before the endpoint, that sample receives one final
core pass at `s=1` and its endpoint head output is returned. Samples already
evaluated at `s=1` do not receive an additional pass.

Active rows are gathered before each core call. Halted rows are removed after
their endpoint pass. `max_loops` is a hard per-sample evaluation cap, including
the endpoint. `min_loops` is the first intermediate check; an early exit at that
check normally costs `min_loops+1` evaluations. `max_loops=1` runs one endpoint.
`min_loops=max_loops` disables early exits and follows the fixed grid.

`loops_used` counts actual core evaluations per sample, including endpoint
passes. `total_core_evaluations` and `mean_loops` derive from those counts.
Residual history is indexed by the ordinary loop and original sample order;
inactive rows contain NaN. Endpoint passes are not residual checks.

Counts exclude the prelude, prediction heads, gather/scatter and other overhead.
They are not FLOPs or latency measurements. Endpoint conditioning preserves
the endpoint prediction task but does not prove quality preservation. Comparing
readouts at different depth coordinates is not a convergence certificate;
the residual depends on the depth grid. Quality and speed advantages remain
unvalidated and require matched baselines and multiple seeds.

Fixed `forward()` evaluates only the final head on its default uniform grid.
`forward_trajectory()` remains the training/diagnostic API for all readouts.

## Boundaries

- `models/`: architecture only.
- `flow/`: objectives and numerical sampling.
- `adaptive/`: policies that alter recurrent-compute allocation.
- `training/`: state management such as EMA/checkpoints.
- `data/`: datasets and preprocessing, with deterministic fixtures first.
- `evaluation/`: metrics and profiling, never training logic.
