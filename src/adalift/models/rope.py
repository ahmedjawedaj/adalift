from __future__ import annotations

import torch


def _rotate_pairs(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to the last dimension of ``x`` using per-token angles."""
    if x.shape[-1] % 2:
        raise ValueError("RoPE dimension must be even")
    pair = x.reshape(*x.shape[:-1], -1, 2)
    x0, x1 = pair.unbind(dim=-1)
    cos = angles.cos()[None, None]
    sin = angles.sin()[None, None]
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos
    return torch.stack([out0, out1], dim=-1).flatten(-2)


def build_2d_rope_angles(
    grid_size: tuple[int, int],
    axis_dim: int,
    *,
    device: torch.device,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return row and column RoPE angles for a flattened 2D token grid.

    ``axis_dim`` is the feature dimension allocated to each axis and must be even.
    The returned angle tensors have shape ``[num_tokens, axis_dim // 2]``.
    """
    if axis_dim % 2:
        raise ValueError("axis_dim must be even")
    gh, gw = grid_size
    rows = torch.arange(gh, device=device).repeat_interleave(gw)
    cols = torch.arange(gw, device=device).repeat(gh)
    pair_count = axis_dim // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(pair_count, device=device, dtype=torch.float32) / pair_count)
    )
    return rows[:, None].float() * inv_freq[None], cols[:, None].float() * inv_freq[None]


def apply_2d_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    grid_size: tuple[int, int],
    theta: float = 10_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply independent rotary embeddings to row/column halves of attention heads.

    Args:
        q, k: ``[batch, heads, tokens, head_dim]``.
    """
    if q.shape != k.shape:
        raise ValueError("q and k shapes must match")
    if q.ndim != 4:
        raise ValueError("q and k must be rank-4")
    head_dim = q.shape[-1]
    if head_dim % 4:
        raise ValueError("head_dim must be divisible by 4 for 2D RoPE")
    expected_tokens = grid_size[0] * grid_size[1]
    if q.shape[-2] != expected_tokens:
        raise ValueError("token count does not match grid_size")

    axis_dim = head_dim // 2
    row_angles, col_angles = build_2d_rope_angles(
        grid_size, axis_dim, device=q.device, theta=theta
    )
    q_row, q_col = q.split(axis_dim, dim=-1)
    k_row, k_col = k.split(axis_dim, dim=-1)
    return (
        torch.cat([_rotate_pairs(q_row, row_angles), _rotate_pairs(q_col, col_angles)], dim=-1),
        torch.cat([_rotate_pairs(k_row, row_angles), _rotate_pairs(k_col, col_angles)], dim=-1),
    )
