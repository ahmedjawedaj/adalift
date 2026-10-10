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

## Verified appendix settings

Source: [LiFT v1 Appendix D](https://arxiv.org/html/2610.05538#A4), Tables 14-16
and Section D.3, checked 2026-10-11.

- B/2 width 768, 12 heads, patch size 2, four prelude blocks.
- B/2-R2 is listed in Table 15 with two core blocks and `K_train=4`.
- AdamW: LR `1e-4`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0`.
- 500 warmup updates then constant LR; 500,000 updates, global batch 256.
- Global gradient norm clip 1; EMA decay **0.999**; seed 42; no label dropout.
- BF16 computation with FP32 parameters and loss accumulation.
- Interior depth draws/sort in FP64, epsilon spacing `1e-6`, output FP32.
- Evaluation: EMA, Euler, no classifier-free guidance, paired noise and labels.

The candidate YAML records verified B/2-R2 architecture/training settings.
Matching those values is not a reproduction claim.

## Items deliberately not overclaimed yet

Exact SpeedrunDiT implementation parity, full model registries, ImageNet cached
latent preprocessing, BF16/distributed paper training, evaluator integration and
analytic FLOP parity remain unverified or unimplemented. These belong to G1/G2.
