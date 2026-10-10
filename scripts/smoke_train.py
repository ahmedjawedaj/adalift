from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from adalift.config import ModelConfig, TrainConfig
from adalift.data.synthetic import SyntheticLatentDataset
from adalift.flow.losses import lift_trajectory_loss
from adalift.models.lift import LiFT
from adalift.training.ema import EMA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny end-to-end LiFT training smoke test.")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("outputs/smoke_metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")
    model_cfg = ModelConfig(
        in_channels=4,
        input_size=16,
        patch_size=2,
        hidden_size=128,
        depth=6,
        prelude_depth=2,
        recurrent_depth=2,
        num_heads=4,
        num_classes=10,
        depth_embed_freq_dim=64,
        time_embed_freq_dim=64,
    )
    train_cfg = TrainConfig(k_train=2, prelude_loss_weight=0.01, learning_rate=2e-4)
    device = torch.device(args.device)
    torch.manual_seed(train_cfg.seed)

    model = LiFT(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate)
    ema = EMA(model, train_cfg.ema_decay)
    dataset = SyntheticLatentDataset(length=max(64, args.batch_size * args.steps), size=16)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    iterator = iter(loader)

    metrics: list[dict[str, float | int]] = []
    model.train()
    for step in range(1, args.steps + 1):
        try:
            latents, labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            latents, labels = next(iterator)
        latents = latents.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        result = lift_trajectory_loss(
            model,
            latents,
            labels,
            k_train=train_cfg.k_train,
            prelude_loss_weight=train_cfg.prelude_loss_weight,
        )
        result.loss.backward()
        if train_cfg.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
        optimizer.step()
        ema.update(model)
        row = {
            "step": step,
            "loss": float(result.loss.detach()),
            "trajectory_loss": float(result.trajectory_loss.detach()),
            "prelude_loss": float(result.prelude_loss.detach()),
        }
        metrics.append(row)
        print(json.dumps(row))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
