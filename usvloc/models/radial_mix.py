from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RadialMixerLayer(nn.Module):
    """One MLP-Mixer layer used by RadialMix.

    The first MLP mixes information along radial bins, and the second MLP
    mixes along the channel dimension. The angular dimension is omitted here
    because AngularGeM has already aggregated angular information.
    """

    def __init__(self, channels: int, radial_bins: int, mlp_ratio: float = 1.0) -> None:
        super().__init__()
        channels = int(channels)
        radial_bins = int(radial_bins)
        radial_hidden = max(1, int(round(float(radial_bins) * float(mlp_ratio))))
        channel_hidden = max(1, int(round(float(channels) * float(mlp_ratio))))

        self.radial_mix = nn.Sequential(
            nn.LayerNorm(radial_bins),
            nn.Linear(radial_bins, radial_hidden),
            nn.GELU(),
            nn.Linear(radial_hidden, radial_bins),
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channel_hidden),
            nn.GELU(),
            nn.Linear(channel_hidden, channels),
        )

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.radial_mix(x)
        x_t = x.transpose(1, 2)
        x_t = x_t + self.channel_mix(x_t)
        return x_t.transpose(1, 2).contiguous()


class RadialMixVPRHead(nn.Module):
    """USVLoc global descriptor head.

    Input [B,C,R] radial features pass through multiple radial/channel mixing
    layers, are projected to ``out_channels x out_rows``, then flattened and
    L2-normalized. The default descriptor size is 512x8=4096.
    """

    def __init__(
        self,
        in_channels: int,
        in_rows: int = 16,
        out_channels: int = 512,
        mix_depth: int = 3,
        mlp_ratio: float = 1.0,
        out_rows: int = 8,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.in_rows = int(in_rows)
        self.out_channels = int(out_channels)
        self.out_rows = int(out_rows)

        self.mix = nn.Sequential(
            *[
                RadialMixerLayer(
                    channels=self.in_channels,
                    radial_bins=self.in_rows,
                    mlp_ratio=float(mlp_ratio),
                )
                for _ in range(max(int(mix_depth), 1))
            ]
        )
        self.channel_proj = nn.Linear(self.in_channels, self.out_channels)
        self.row_proj = nn.Linear(self.in_rows, self.out_rows)

        for module in (self.channel_proj, self.row_proj):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def output_dim(self) -> int:
        return int(self.out_channels * self.out_rows)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,C,R] radial features, got {tuple(x.shape)}")
        if int(x.shape[1]) != self.in_channels or int(x.shape[2]) != self.in_rows:
            raise ValueError(
                f"RadialMixVPRHead input shape mismatch: expected [B,{self.in_channels},{self.in_rows}], got {tuple(x.shape)}"
            )
        x = self.mix(x)
        x = x.transpose(1, 2)
        x = self.channel_proj(x)
        x = x.transpose(1, 2)
        x = self.row_proj(x)
        return F.normalize(x.flatten(1), p=2, dim=1)
