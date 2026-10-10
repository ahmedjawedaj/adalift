from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from adalift.models.rope import apply_2d_rope


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale[:, None]) + shift[:, None]


class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        qk_norm: bool = True,
        rope_theta: float = 10_000.0,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qk_norm = qk_norm
        self.rope_theta = rope_theta
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor, grid_size: tuple[int, int]) -> torch.Tensor:
        bsz, tokens, hidden = x.shape
        qkv = self.qkv(x).reshape(bsz, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.qk_norm:
            q = F.rms_norm(q, (self.head_dim,))
            k = F.rms_norm(k, (self.head_dim,))
        q, k = apply_2d_rope(q, k, grid_size, self.rope_theta)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(bsz, tokens, hidden)
        return self.proj(out)


class DiTBlock(nn.Module):
    """DiT-style block with RMSNorm, QK norm, RoPE and adaLN-Zero gates."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        *,
        qk_norm: bool = True,
        rope_theta: float = 10_000.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.attn = SelfAttention(
            hidden_size,
            num_heads,
            qk_norm=qk_norm,
            rope_theta=rope_theta,
        )
        self.norm2 = nn.RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        grid_size: tuple[int, int],
    ) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.ada(condition).chunk(6, dim=-1)
        x = x + gate_a[:, None] * self.attn(
            modulate(self.norm1(x), shift_a, scale_a), grid_size
        )
        x = x + gate_m[:, None] * self.mlp(modulate(self.norm2(x), shift_m, scale_m))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, patch_size: int) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.proj = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        condition: torch.Tensor,
        grid_size: tuple[int, int],
    ) -> torch.Tensor:
        shift, scale = self.ada(condition).chunk(2, dim=-1)
        tokens = self.proj(modulate(self.norm(tokens), shift, scale))
        return self.unpatchify(tokens, grid_size)

    def unpatchify(self, tokens: torch.Tensor, grid_size: tuple[int, int]) -> torch.Tensor:
        bsz, num_tokens, _ = tokens.shape
        gh, gw = grid_size
        if num_tokens != gh * gw:
            raise ValueError("token count does not match grid")
        p = self.patch_size
        x = tokens.reshape(bsz, gh, gw, p, p, self.out_channels)
        x = torch.einsum("bhwpqc->bchpwq", x)
        return x.reshape(bsz, self.out_channels, gh * p, gw * p)
