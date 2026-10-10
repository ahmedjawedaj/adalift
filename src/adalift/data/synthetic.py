from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SyntheticLatentDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Deterministic structured latent data for smoke tests and CI.

    This is not a scientific benchmark. It exists so every contributor can validate the full
    training path without downloading data or requiring a GPU.
    """

    def __init__(
        self,
        *,
        length: int = 256,
        channels: int = 4,
        size: int = 16,
        num_classes: int = 10,
        seed: int = 1234,
    ) -> None:
        self.length = length
        self.channels = channels
        self.size = size
        self.num_classes = num_classes
        generator = torch.Generator().manual_seed(seed)
        self.class_templates = (
            torch.randn(num_classes, channels, size, size, generator=generator) * 0.5
        )
        self.noise_seeds = torch.randint(0, 2**31 - 1, (length,), generator=generator)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        label = index % self.num_classes
        generator = torch.Generator().manual_seed(int(self.noise_seeds[index]))
        noise = torch.randn(self.channels, self.size, self.size, generator=generator) * 0.05
        return self.class_templates[label] + noise, torch.tensor(label, dtype=torch.long)
