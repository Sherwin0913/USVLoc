from __future__ import annotations

import torch
import torch.nn as nn


class AngularGeMPool(nn.Module):
    """Angular GeM pooling.

    The input is [B,C,R,A] polar features, where A is the angular dimension.
    This module aggregates only along the angular axis and outputs [B,C,R], so
    the global descriptor retains radial layout information.
    """

    def __init__(self, p_init: float = 3.0, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * float(p_init))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,R,A] polar features, got {tuple(x.shape)}")
        p = torch.clamp(self.p, min=1.0)
        return x.clamp(min=self.eps).pow(p).mean(dim=-1).pow(1.0 / p)
