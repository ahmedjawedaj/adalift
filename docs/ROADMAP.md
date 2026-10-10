# Roadmap

## Milestone G0 — Core correctness

- [x] LiFT prelude/core/head decomposition
- [x] recurrent boundary reinjection
- [x] continuous depth conditioning
- [x] random sorted training-depth coordinates
- [x] stop-gradient anchor trajectory targets
- [x] auxiliary prelude loss
- [x] fixed-depth Euler sampling
- [x] residual adaptive-depth baseline
- [x] CPU smoke test and unit suite
- [ ] verify architecture hyperparameters line-by-line against official code when released

## Milestone G1 — Mini-LiFT falsification

- [ ] choose CIFAR latent or ImageNet-64 benchmark
- [ ] deterministic train/eval pipeline
- [ ] dense baseline at matched executed depth
- [ ] `K_train=2`, evaluate `K_inf={1,2,4,8,16}`
- [ ] >= 3 seeds
- [ ] publish curves and failure analysis

## Milestone G2 — Faithful ImageNet reproduction

- [ ] SD-VAE latent preprocessing
- [ ] verify B/2 and L/2 exact dimensions
- [ ] distributed training
- [ ] EMA sampling
- [ ] FID-50K/sFID/IS/precision/recall
- [ ] FLOP accounting
- [ ] latency/VRAM profiling

## Milestone G3 — AdaLiFT

- [x] residual-based early exit baseline
- [ ] timestep-specific residual thresholds
- [ ] learned halting head
- [ ] compute-regularized halting objective
- [ ] learned/nonuniform depth coordinates
- [ ] matched-FLOP comparison against fixed K

## Milestone G4 — Time/depth compute allocation

- [ ] Pareto sweep over `(T, K)`
- [ ] adaptive per-timestep K
- [ ] sample-adaptive K
- [ ] wall-clock and energy measurements

## Milestone G5 — Public research release

- [ ] checkpoints
- [ ] reproducibility bundle
- [ ] model cards
- [ ] technical report/preprint
- [ ] contributor benchmark board
