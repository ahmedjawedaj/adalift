# AdaLiFT

**Adaptive recurrent depth for flow transformers.**

AdaLiFT is a clean-room PyTorch research implementation inspired by **LiFT: Loop Flow Transformers** (Derakhshani et al., 2026), plus an extension path for adaptive test-time depth. The project is designed for reproducible research, careful ablations, and community contribution rather than one-off benchmark scripts.

> Status: **research alpha, RC2 for independent review**. Core implementation and
> synthetic verification are available. No ImageNet metric reproduction or
> adaptive quality/latency advantage is claimed. RC2 starts from the audited
> local snapshot; the unavailable RC1 ZIP was never verified.

## Why this project

LiFT separates generative **time** from recurrent **depth**: a shared DiT core can be applied repeatedly inside each flow-model evaluation, allowing inference compute to increase without adding parameters. AdaLiFT asks the next question: **can each sample and flow step decide how much recurrent computation it actually needs?**

## Implemented now

- DiT-style latent patch transformer with RMSNorm, QK normalization, 2D rotary position embeddings, and adaLN-Zero-style conditioning.
- `Prelude -> shared recurrent core -> prediction head` LiFT architecture.
- Input-state reinjection at recurrent boundaries: `RMSNorm(h) + h0`.
- Random sorted continuous depth coordinates during training.
- LiFT trajectory targets from a stop-gradient prelude anchor to the flow-matching target.
- Auxiliary prelude loss.
- Fixed-depth inference and endpoint-preserving residual-adaptive inference with active-row compaction.
- Actual per-sample core-evaluation counts, including endpoint passes.
- Euler flow sampler.
- EMA and atomic checkpoint helpers.
- Smoke-training entry point on deterministic synthetic latent data.
- Unit tests and GitHub Actions CI.
- Repo-local Agent Skills for Codex and mirrored Claude Code project skills.

## Research milestones

1. **G0 — implementation correctness:** unit tests + toy convergence.
2. **G1 — mini-LiFT diagnostic:** inspect `K_inf=4/8/16`; tiny-model failure is not a project rejection gate.
3. **G2 — faithful reproduction:** latent ImageNet subset, then ImageNet-256.
4. **G3 — AdaLiFT:** residual and learned halting at matched FLOPs.
5. **G4 — compute allocation:** optimize flow steps `T` vs recurrent depth `K`.
6. **G5 — release:** reproducibility bundle, checkpoints, FID artifacts, technical report.

See [`docs/ROADMAP.md`](docs/ROADMAP.md) and [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python scripts/smoke_train.py --steps 10 --batch-size 4
adalift-smoke --config configs/mini_lift.yaml --steps 3 --batch-size 2
make verify
python scripts/verify_distribution.py
```

Minimal model usage:

```python
import torch
from adalift.config import ModelConfig
from adalift.models.lift import LiFT

cfg = ModelConfig(
    in_channels=4,
    input_size=16,
    patch_size=2,
    hidden_size=128,
    depth=6,
    prelude_depth=2,
    recurrent_depth=2,
    num_heads=4,
    num_classes=10,
)
model = LiFT(cfg)

x = torch.randn(2, 4, 16, 16)
t = torch.rand(2)
y = torch.tensor([1, 2])
velocity = model(x, t, y, num_loops=4)
print(velocity.shape)
```

Adaptive outputs always include a final `s=1` core pass. `max_loops` is the
hard evaluation budget; `loops_used` includes the endpoint pass. Counts measure
core evaluations, not FLOPs or wall-clock. Samplers accept common `initial_noise`,
select evaluation mode temporarily, and restore the caller's module modes.
See [architecture](docs/ARCHITECTURE.md) for budget and residual semantics.

`configs/lift_b2_r2_candidate.yaml` records the B/2-R2 architecture and optimizer
fields from the paper's Appendix D (width, heads, prelude and core depth, `K_train=4`,
AdamW settings, EMA, warmup, clipping). It does not encode the paper's batch size,
update count, data or precision, and the smoke runner still uses synthetic FP32
data. Release review instructions and
remaining limitations are in [REVIEW_HANDOFF.md](docs/REVIEW_HANDOFF.md).

## Project design

```text
src/adalift/
  config.py
  models/         # transformer blocks, embeddings, LiFT
  flow/           # training objective and samplers
  adaptive/       # adaptive-depth policies
  training/       # EMA and checkpointing
  data/           # deterministic research datasets
  evaluation/     # metrics/profiling helpers
```

## Agent-native development

- **Codex:** reads [`AGENTS.md`](AGENTS.md) and repo-local skills under `.agents/skills/`.
- **Claude Code:** reads [`CLAUDE.md`](CLAUDE.md) and project skills under `.claude/skills/`.
- The canonical skills live in `.agents/skills/`; run `make sync-skills` after changing one.

## Scientific integrity

We do not label a result "reproduced" because a loss curve looks plausible. Claims require a recorded config, code commit, seed, hardware/software environment, sample count, metric implementation, and comparison protocol. See `docs/REPRODUCIBILITY.md`.

## Relationship to LiFT

This is an independent implementation based on the published paper and public descriptions. It does not copy unreleased author code. When official LiFT code is released, we will add compatibility tests and document behavioral differences rather than silently replacing this implementation.

Paper: <https://arxiv.org/abs/2610.05538>

Related work and provenance limitations: [acknowledgements](docs/ACKNOWLEDGEMENTS.md).

## Contributing

Please read [`CONTRIBUTING.md`](CONTRIBUTING.md), [`GOVERNANCE.md`](GOVERNANCE.md), and the research workflow in `.agents/skills/research-reproduction/SKILL.md`.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
