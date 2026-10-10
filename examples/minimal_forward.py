import torch

from adalift.config import ModelConfig
from adalift.models import LiFT

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
    depth_embed_freq_dim=64,
    time_embed_freq_dim=64,
)
model = LiFT(cfg)
x = torch.randn(2, 4, 16, 16)
t = torch.rand(2)
y = torch.tensor([1, 2])
print(model(x, t, y, num_loops=4).shape)
