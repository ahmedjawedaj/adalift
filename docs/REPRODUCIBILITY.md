# Reproducibility Standard

A run is not a reproduction merely because it trains or produces plausible samples.

## Required metadata for every reported experiment

- git commit SHA and dirty/clean status
- model and training config
- random seed(s)
- dataset identity, preprocessing, and split hash/version
- tokenizer/VAE identity and precision
- hardware model/count
- CUDA, PyTorch, and driver versions
- precision mode and TF32 setting
- global batch size and gradient accumulation
- optimizer, learning rate, weight decay
- total optimizer updates and data-token budget
- EMA settings
- `K_train`, `K_inf`, depth-coordinate sampling rule
- integration method and number of flow steps `T`
- sample count and evaluation code/version

## Claim levels

### Smoke verified
Code executes end to end and tests pass. No scientific claim.

### Behavior reproduced
A qualitative or local property is observed with a documented small setup, e.g. useful `K_inf > K_train`. This does not imply ImageNet headline reproduction.

### Metric reproduced
A paper metric is matched within a predeclared tolerance under materially equivalent data, architecture, training budget, and evaluation protocol.

### Extension validated
The proposed extension beats predeclared baselines at matched compute/parameters across at least three seeds and includes negative/failure cases.

## First falsification gate

Train a small model at `K_train=2`. Evaluate the same checkpoint at `K_inf in {1,2,4,8,16}` using common initial noise and labels. Continue to expensive scaling only if extra recurrent depth is useful beyond the training depth and the result is stable across seeds.

## Artifact policy

Large datasets/checkpoints stay outside git. Release immutable hashes and machine-readable metric files. Public weights should use `safetensors` when practical.
