import torch

from adalift.models.rope import apply_2d_rope


def test_rope_preserves_shape_and_norm() -> None:
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn_like(q)
    q2, k2 = apply_2d_rope(q, k, (4, 4))
    assert q2.shape == q.shape
    assert k2.shape == k.shape
    torch.testing.assert_close(q.norm(dim=-1), q2.norm(dim=-1), rtol=1e-5, atol=1e-5)
