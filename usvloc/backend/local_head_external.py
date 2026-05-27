"""External local descriptor head for USVLoc backend RANSAC.

The frozen USVLoc checkpoint exposes ``cartesian_features`` with shape
``[B,128,26,26]``. This standalone head upsamples those features into dense
``[B,128,201,201]`` L2-normalized descriptors without changing the main model
forward or checkpoint format.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExternalLocalHead(nn.Module):
    def __init__(
        self,
        in_dim: int = 128,
        mid: int = 128,
        out_dim: int = 128,
        out_size: int = 201,
    ) -> None:
        super().__init__()
        self.out_size = int(out_size)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(int(in_dim), int(mid), kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, int(mid)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(mid), int(mid), kernel_size=3, padding=1),
            nn.GroupNorm(8, int(mid)),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(int(mid), int(mid), kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, int(mid)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(mid), int(mid), kernel_size=3, padding=1),
            nn.GroupNorm(8, int(mid)),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(int(mid), int(mid), kernel_size=3, padding=1),
            nn.GroupNorm(8, int(mid)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(mid), int(out_dim), kernel_size=1),
        )

    def forward(self, cartesian_features: torch.Tensor) -> torch.Tensor:
        if cartesian_features.ndim != 4:
            raise ValueError(f"Expected cartesian features [B,C,H,W], got {tuple(cartesian_features.shape)}")
        x = self.up1(cartesian_features)
        x = self.up2(x)
        x = F.interpolate(
            x,
            size=(self.out_size, self.out_size),
            mode="bilinear",
            align_corners=False,
        )
        x = self.head(x)
        return F.normalize(x, p=2, dim=1)
