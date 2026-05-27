from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class CartesianToPolar(nn.Module):
    """Differentiable resampling from Cartesian feature maps to polar feature maps.

    Input [B,C,H,W] is converted to [B,C,R,A]. Each polar sample point
    ``(rho, theta)`` is mapped back to Cartesian coordinates and interpolated
    with ``grid_sample``. Samples outside the valid feature map follow the
    PyTorch ``grid_sample`` default zero-padding behavior.
    """

    def __init__(
        self,
        radial_bins: int = 16,
        angular_bins: int = 64,
        center_xy: Sequence[float] | None = None,
        radius_max: float | None = None,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.radial_bins = int(radial_bins)
        self.angular_bins = int(angular_bins)
        self.center_xy = None if center_xy is None else (float(center_xy[0]), float(center_xy[1]))
        self.radius_max = None if radius_max is None else float(radius_max)
        self.align_corners = bool(align_corners)

    def _to_normalized(self, coords: torch.Tensor, size: int) -> torch.Tensor:
        if self.align_corners:
            if size <= 1:
                return torch.zeros_like(coords)
            return (2.0 * coords / float(size - 1)) - 1.0
        return (2.0 * (coords + 0.5) / float(size)) - 1.0

    def _build_grid(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        center_x = float(width) / 2.0 if self.center_xy is None else float(self.center_xy[0])
        center_y = float(height) / 2.0 if self.center_xy is None else float(self.center_xy[1])
        radius_max = self.radius_max
        if radius_max is None:
            radius_max = math.sqrt(center_x * center_x + center_y * center_y)

        radii = torch.linspace(0.0, float(radius_max), steps=self.radial_bins, device=device, dtype=dtype)
        angles = torch.linspace(0.0, 2.0 * math.pi, steps=self.angular_bins + 1, device=device, dtype=dtype)[:-1]
        rr = radii[:, None]
        tt = angles[None, :]

        x = center_x + rr * torch.cos(tt)
        y = center_y + rr * torch.sin(tt)
        # grid_sample expects normalized coordinates in [-1, 1] with the last dimension ordered as (x, y).
        grid_x = self._to_normalized(x, width)
        grid_y = self._to_normalized(y, height)
        return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W] cartesian features, got {tuple(x.shape)}")
        grid = self._build_grid(
            height=int(x.shape[2]),
            width=int(x.shape[3]),
            device=x.device,
            dtype=x.dtype,
        ).expand(int(x.shape[0]), -1, -1, -1)
        return F.grid_sample(x, grid, mode="bilinear", align_corners=self.align_corners)


class PolarMixStyle(nn.Module):
    """MixStyle on polar features.

    During training, randomly swap and mix channel means and variances across
    batch samples to simulate BEV style shifts across domains. At inference, the
    original features are returned without extra overhead.
    """

    def __init__(self, enabled: bool = False, p: float = 0.5, alpha: float = 0.1, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.beta = torch.distributions.Beta(self.alpha, self.alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled or not self.training:
            return x
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,R,A] polar features, got {tuple(x.shape)}")
        if int(x.shape[0]) < 2:
            return x
        if torch.rand(1, device=x.device).item() > self.p:
            return x

        mu = x.mean(dim=(2, 3), keepdim=True).detach()
        sigma = torch.sqrt(x.var(dim=(2, 3), keepdim=True, unbiased=False) + self.eps).detach()
        x_normed = (x - mu) / sigma

        perm = torch.randperm(int(x.shape[0]), device=x.device)
        mu_perm = mu[perm]
        sigma_perm = sigma[perm]

        lam = self.beta.sample((int(x.shape[0]), 1, 1, 1)).to(device=x.device, dtype=x.dtype)
        mu_mix = lam * mu + (1.0 - lam) * mu_perm
        sigma_mix = lam * sigma + (1.0 - lam) * sigma_perm
        return x_normed * sigma_mix + mu_mix


class CircularConvBlock(nn.Module):
    """Angular circular convolution block.

    The angular dimension of a polar map has periodic boundaries, so 0 and 360
    degrees should be adjacent. Therefore, only the angular dimension uses
    ``mode='circular'`` padding, while the radial dimension keeps standard
    zero padding.
    """

    def __init__(self, channels: int = 128, kernel_size: int = 3) -> None:
        super().__init__()
        channels = int(channels)
        kernel_size = int(kernel_size)
        if kernel_size % 2 == 0:
            raise ValueError(f"CircularConvBlock requires odd kernel_size, got {kernel_size}")
        theta_pad = kernel_size // 2
        radial_pad = kernel_size // 2
        self.theta_pad = int(theta_pad)
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=(radial_pad, 0),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.theta_pad, self.theta_pad, 0, 0), mode="circular")
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)
