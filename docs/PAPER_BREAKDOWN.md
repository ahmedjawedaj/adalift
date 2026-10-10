# LiFT Paper Breakdown

Source: Derakhshani et al., **LiFT: Loop Flow Transformers**, arXiv:2610.05538 (2026).

## Core claim

A shared recurrent DiT core can support adjustable inference-time depth when intermediate recurrent predictions are supervised along a continuous reference path in prediction space rather than each being forced directly to the final flow target.

## Architecture facts currently encoded

- DiT backbone is partitioned into a prelude, recurrent core, and prediction-only coda/head.
- Main paper comparisons use four prelude blocks and a zero-transformer-block coda.
- Outer conditioning is generative time plus class label.
- Recurrent core additionally receives continuous depth-coordinate conditioning.
- Before each recurrent application, the previous state is normalized and the prelude state is reinjected.
- Training reads the prelude and every recurrent state; inference only needs the final readout.

## Training facts currently encoded

- Linear flow path between Gaussian noise and data latent.
- Flow target is `x1 - x0`.
- Prelude prediction is stop-gradient only on the target/anchor side.
- `K-1` interior depth coordinates are sampled uniformly and sorted; final depth is `1`.
- Recurrent prediction `u_k` regresses to `(1-s_k) * stopgrad(u0) + s_k * u*`.
- Auxiliary prelude regression weight defaults to `0.01` in the paper's main experiments.

## Items deliberately not overclaimed yet

The current repo has not yet locked exact SpeedrunDiT implementation details, full B/L/XL model registries, ImageNet latent preprocessing, analytic FLOP parity, or paper-exact optimizer/runtime settings beyond the published high-level values. These belong to G1/G2 and must be verified against the paper appendix and official code when available.
