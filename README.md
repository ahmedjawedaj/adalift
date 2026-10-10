# AdaLiFT

**Adaptive recurrent depth for flow transformers.**

AdaLiFT is a clean-room PyTorch research implementation inspired by **LiFT: Loop Flow Transformers** (Derakhshani et al., 2026), plus an extension path for adaptive test-time depth. The project is designed for reproducible research, careful ablations, and community contribution rather than one-off benchmark scripts.

> Status: **research alpha**. The core LiFT trajectory objective, recurrent architecture, Euler flow sampler, residual-based adaptive halting, checkpoint/EMA utilities, tests, CI, and agent workflows are implemented. ImageNet-scale reproduction is intentionally gated behind smaller validation experiments.

## Why this project

LiFT separates generative **time** from recurrent **depth**: a shared DiT core can be applied repeatedly inside each flow-model evaluation, allowing inference compute to increase without adding parameters. AdaLiFT asks the next question: **can each sample and flow step decide how much recurrent computation it actually needs?**

## Implemented now

- DiT-style latent patch transformer with RMSNorm, QK normalization, 2D rotary position embeddings, and adaLN-Zero-style conditioning.
- `Prelude -> shared recurrent core -> prediction head` LiFT architecture.
- Input-state reinjection at recurrent boundaries: `RMSNorm(h) + h0`.
- Random sorted continuous depth coordinates during training.
- LiFT trajectory targets from a stop-gradient prelude anchor to the flow-matching target.
- Auxiliary prelude loss.
- Fixed-depth and residual-adaptive recurrent inference.
- Euler flow sampler.
- EMA and atomic checkpoint helpers.
- Smoke-training entry point on deterministic synthetic latent data.
- Unit tests and GitHub Actions CI.
- Repo-local Agent Skills for Codex and mirrored Claude Code project skills.

## Research milestones

1. **G0 — implementation correctness:** unit tests + toy convergence.
2. **G1 — mini-LiFT:** train at `K_train=2`; verify useful inference at `K_inf=4/8/16`.
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

## Contributing

Please read [`CONTRIBUTING.md`](CONTRIBUTING.md), [`GOVERNANCE.md`](GOVERNANCE.md), and the research workflow in `.agents/skills/research-reproduction/SKILL.md`.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
