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

## Diagnostic and scale gates

Train a small model at `K_train=2` and evaluate `K_inf in {1,2,4,8,16}` with
common initial noise and labels as a diagnostic. A tiny-model failure to
extrapolate does not reject LiFT or the adaptive hypothesis. Test sufficiently
capable paper-scale configurations before making a scale-related conclusion.
Adaptive extensions require at least three seeds, endpoint outputs, and fixed
baselines at matched measured compute. Do not interpolate a quality curve and
call it a measured matched-budget baseline.

## Implemented execution and resume scope

`adalift-smoke --config configs/mini_lift.yaml` reads the actual YAML and writes
a JSON run record with resolved configurations, source identity when available,
seed, synthetic dataset identity, precision and environment. It is a synthetic
FP32 smoke runner, including when a paper-scale candidate config is supplied.
It does not implement ImageNet data, BF16 paper training, distributed execution,
or FID evaluation. It implements AdamW and linear warmup followed by constant LR.

`save_checkpoint(..., capture_rng=True, generator=...)` records Torch CPU/CUDA,
Python random, and optional generator state. Pass `restore_rng=True` to restore.
Requested optimizer/EMA/RNG states must exist. Store model/train config as plain
metadata. Dataset progress, data-loader generators, NumPy/external RNG state,
hardware equivalence and deterministic kernels remain caller responsibilities.
RNG restoration alone does not guarantee exact full training resume.

## Artifact policy

Large datasets/checkpoints stay outside git. Release immutable hashes and machine-readable metric files. Public weights should use `safetensors` when practical.
