from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolarSelfAttention(nn.Module):
    """Polar self-attention module.

    Input [B,C,R,A] is flattened into R*A tokens. P-RoPE splits ``head_dim``
    into radial and angular parts: the radial part uses standard RoPE
    frequencies, while the angular part uses periodic angular frequencies so
    attention can explicitly encode relative positions on the polar lattice.
    """

    def __init__(
        self,
        channels: int = 128,
        num_heads: int = 4,
        rope_mode: str = "none",
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        residual_scale_init: float = 0.1,
        norm_eps: float = 1.0e-5,
        enabled: bool = False,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.rope_mode = str(rope_mode).lower().replace("-", "_")
        self.attn_dropout = float(attn_dropout)
        self.proj_dropout = nn.Dropout(float(proj_dropout))

        if not self.enabled:
            self.norm = nn.Identity()
            self.qkv = nn.Identity()
            self.out_proj = nn.Identity()
            self.residual_scale = None
            return

        if self.channels % self.num_heads != 0:
            raise ValueError(
                f"PolarSelfAttention requires channels divisible by num_heads, got channels={self.channels}, "
                f"num_heads={self.num_heads}"
            )
        self.head_dim = self.channels // self.num_heads
        if self.rope_mode in {"polar", "p_rope", "prope"} and self.head_dim % 4 != 0:
            raise ValueError(
                "PolarSelfAttention with P-RoPE requires head_dim % 4 == 0, "
                f"got head_dim={self.head_dim}"
            )

        self.norm = nn.LayerNorm(self.channels, eps=float(norm_eps))
        self.qkv = nn.Linear(self.channels, self.channels * 3, bias=False)
        self.out_proj = nn.Linear(self.channels, self.channels, bias=False)
        self.residual_scale = nn.Parameter(torch.ones(1) * float(residual_scale_init))

    def _apply_axis_rope(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        sin = sin[None, None, :, :]
        cos = cos[None, None, :, :]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        return torch.stack([out_even, out_odd], dim=-1).flatten(-2)

    def _apply_polar_rope(self, x: torch.Tensor, radial_bins: int, angular_bins: int) -> torch.Tensor:
        if self.rope_mode not in {"polar", "p_rope", "prope"}:
            return x

        _, _, _, head_dim = x.shape
        radial_dim = head_dim // 2
        angular_dim = head_dim - radial_dim
        x_radial = x[..., :radial_dim]
        x_angular = x[..., radial_dim:]

        device = x.device
        dtype = x.dtype
        # Radial positions are non-periodic, so use the common 10000 base frequency from transformer RoPE.
        r_pos = torch.arange(radial_bins, device=device, dtype=dtype)
        r_freq = 1.0 / (
            10000.0 ** (torch.arange(0, radial_dim, 2, device=device, dtype=dtype) / float(max(radial_dim, 1)))
        )
        radial_angles = torch.outer(r_pos, r_freq)

        # Angular positions are periodic; scale frequencies by 2pi/A to keep the endpoints continuous.
        a_pos = torch.arange(angular_bins, device=device, dtype=dtype)
        a_freq = (2.0 * math.pi / float(angular_bins)) * torch.arange(
            1,
            (angular_dim // 2) + 1,
            device=device,
            dtype=dtype,
        )
        angular_angles = torch.outer(a_pos, a_freq)

        r_idx, a_idx = torch.meshgrid(
            torch.arange(radial_bins, device=device),
            torch.arange(angular_bins, device=device),
            indexing="ij",
        )
        r_idx = r_idx.reshape(-1)
        a_idx = a_idx.reshape(-1)

        radial_sin = torch.sin(radial_angles)[r_idx]
        radial_cos = torch.cos(radial_angles)[r_idx]
        angular_sin = torch.sin(angular_angles)[a_idx]
        angular_cos = torch.cos(angular_angles)[a_idx]

        x_radial = self._apply_axis_rope(x_radial, radial_sin, radial_cos)
        x_angular = self._apply_axis_rope(x_angular, angular_sin, angular_cos)
        return torch.cat([x_radial, x_angular], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,R,A] polar features, got {tuple(x.shape)}")

        batch_size, channels, radial_bins, angular_bins = x.shape
        if channels != self.channels:
            raise ValueError(f"Expected channels={self.channels}, got {channels}")

        num_tokens = int(radial_bins * angular_bins)
        # [B,C,R,A] -> [B,R*A,C], where each polar grid cell is one attention token.
        tokens = x.permute(0, 2, 3, 1).reshape(batch_size, num_tokens, channels)
        tokens = self.norm(tokens)

        qkv = self.qkv(tokens).reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.rope_mode in {"polar", "p_rope", "prope"}:
            # Apply P-RoPE only to q/k; v keeps the original content features.
            q = self._apply_polar_rope(q, radial_bins=radial_bins, angular_bins=angular_bins)
            k = self._apply_polar_rope(k, radial_bins=radial_bins, angular_bins=angular_bins)

        dropout_p = self.attn_dropout if self.training else 0.0
        if hasattr(F, "scaled_dot_product_attention"):
            attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        else:
            scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            weights = torch.softmax(scores, dim=-1)
            if dropout_p > 0.0:
                weights = F.dropout(weights, p=dropout_p, training=True)
            attn_out = weights @ v
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, num_tokens, channels)
        attn_out = self.proj_dropout(self.out_proj(attn_out))
        attn_out = attn_out.reshape(batch_size, radial_bins, angular_bins, channels).permute(0, 3, 1, 2)
        return x + self.residual_scale * attn_out
