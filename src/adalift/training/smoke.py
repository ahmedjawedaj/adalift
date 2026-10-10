from __future__ import annotations

import argparse
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from adalift import __version__
from adalift.config import ModelConfig, SamplingConfig, TrainConfig, load_yaml
from adalift.data.synthetic import SyntheticLatentDataset
from adalift.flow.losses import lift_trajectory_loss
from adalift.models.lift import LiFT
from adalift.training.ema import EMA


def read_run_config(path: Path | None) -> tuple[ModelConfig, TrainConfig, SamplingConfig]:
    if path is None:
        model = ModelConfig(
            in_channels=4,
            input_size=16,
            hidden_size=128,
            depth=6,
            prelude_depth=2,
            recurrent_depth=2,
            num_heads=4,
            num_classes=10,
            time_embed_freq_dim=64,
            depth_embed_freq_dim=64,
        )
        train = TrainConfig(learning_rate=2e-4, warmup_steps=0)
        sampling = SamplingConfig()
    else:
        values = load_yaml(path)
        unknown = set(values) - {"model", "training", "sampling"}
        if unknown:
            raise ValueError(f"unknown config sections: {sorted(unknown)}")
        model = ModelConfig(**values.get("model", {}))
        train = TrainConfig(**values.get("training", {}))
        sampling = SamplingConfig(**values.get("sampling", {}))
    model.validate()
    train.validate()
    sampling.validate()
    return model, train, sampling


def _git_identity() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout
        return {"commit": commit, "dirty": bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic LiFT smoke training; no benchmark claims."
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("outputs/smoke_metrics.json"))
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error("--steps and --batch-size must be >= 1")
    model_cfg, train_cfg, sample_cfg = read_run_config(args.config)
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.manual_seed(train_cfg.seed)
    generator = torch.Generator(device=device).manual_seed(train_cfg.seed)
    order_generator = torch.Generator().manual_seed(train_cfg.seed)
    model = LiFT(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        betas=train_cfg.adam_betas,
        eps=train_cfg.adam_eps,
        weight_decay=train_cfg.weight_decay,
    )
    ema = EMA(model, train_cfg.ema_decay)
    dataset = SyntheticLatentDataset(
        length=max(64, args.batch_size * args.steps),
        channels=model_cfg.in_channels,
        size=model_cfg.input_size,
        num_classes=model_cfg.num_classes,
        seed=train_cfg.seed,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, generator=order_generator
    )
    iterator = iter(loader)
    metrics: list[dict[str, float | int]] = []
    model.train()
    for step in range(1, args.steps + 1):
        try:
            latents, labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            latents, labels = next(iterator)
        warmup = min(1.0, step / train_cfg.warmup_steps) if train_cfg.warmup_steps else 1.0
        for group in optimizer.param_groups:
            group["lr"] = train_cfg.learning_rate * warmup
        optimizer.zero_grad(set_to_none=True)
        result = lift_trajectory_loss(
            model,
            latents.to(device),
            labels.to(device),
            k_train=train_cfg.k_train,
            prelude_loss_weight=train_cfg.prelude_loss_weight,
            generator=generator,
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
    record = {
        "evidence_level": "smoke verified",
        "version": __version__,
        "git": _git_identity(),
        "config": {
            "model": asdict(model_cfg),
            "training": asdict(train_cfg),
            "sampling": asdict(sample_cfg),
        },
        "dataset": {"type": "synthetic", "seed": train_cfg.seed, "length": len(dataset)},
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "platform": platform.platform(),
            "device": str(device),
            "cuda": torch.version.cuda,
            "precision": "float32",
        },
        "batch_size": args.batch_size,
        "steps": args.steps,
        "metrics": metrics,
        "limitations": "Synthetic smoke only; no ImageNet preprocessing, distributed training or FID.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
