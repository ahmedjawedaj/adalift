import torch

from adalift.data import SyntheticLatentDataset
from adalift.evaluation import profile_parameters
from adalift.models.embeddings import LabelEmbedding


def test_synthetic_dataset_is_deterministic() -> None:
    a = SyntheticLatentDataset(length=20, size=8, seed=99)
    b = SyntheticLatentDataset(length=20, size=8, seed=99)
    xa, ya = a[7]
    xb, yb = b[7]
    torch.testing.assert_close(xa, xb)
    assert ya.item() == yb.item()
    assert len(a) == 20


def test_parameter_profile() -> None:
    layer = LabelEmbedding(10, 16)
    profile = profile_parameters(layer)
    assert profile.total_parameters > 0
    assert profile.trainable_parameters == profile.total_parameters


def test_label_embedding_dropout_path() -> None:
    layer = LabelEmbedding(3, 8, dropout_prob=1.0)
    layer.train()
    out = layer(torch.tensor([0, 1, 2]))
    assert out.shape == (3, 8)
