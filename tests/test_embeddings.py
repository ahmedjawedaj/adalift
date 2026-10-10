import torch

from adalift.models.embeddings import PatchEmbed2d, sinusoidal_embedding


def test_sinusoidal_embedding_shape() -> None:
    values = torch.tensor([0.0, 0.5, 1.0])
    out = sinusoidal_embedding(values, 7)
    assert out.shape == (3, 7)
    assert torch.isfinite(out).all()


def test_patch_embed_shape() -> None:
    layer = PatchEmbed2d(4, 32, 2)
    x = torch.randn(2, 4, 16, 16)
    tokens, grid = layer(x)
    assert tokens.shape == (2, 64, 32)
    assert grid == (8, 8)
