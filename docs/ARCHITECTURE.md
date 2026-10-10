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

For `K_train > 1`, sample `K_train - 1` interior depth coordinates independently from `U(0,1)`, sort them, and set the final coordinate to `1`. Each recurrent readout is supervised against `ubar(s_k)`.

The total loss is the average recurrent trajectory MSE plus a small auxiliary prelude MSE weighted by `lambda`.

## Adaptive-depth extension

`LiFT.forward_adaptive` is explicitly an AdaLiFT extension, not part of the original paper. The first baseline computes normalized velocity-change residuals across recurrent passes and exits samples after the residual falls below a threshold. This baseline is intentionally simple so learned halting must beat a strong training-free comparator.

## Boundaries

- `models/`: architecture only.
- `flow/`: objectives and numerical sampling.
- `adaptive/`: policies that alter recurrent-compute allocation.
- `training/`: state management such as EMA/checkpoints.
- `data/`: datasets and preprocessing, with deterministic fixtures first.
- `evaluation/`: metrics and profiling, never training logic.
