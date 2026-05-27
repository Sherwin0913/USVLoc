from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import PolarSelfAttention
from .backbone import TruncatedResNetEncoder
from .polar import CartesianToPolar, CircularConvBlock, PolarMixStyle
from .pooling import AngularGeMPool
from .radial_mix import RadialMixVPRHead


class USVLoc(nn.Module):
    """Main USVLoc network.

    The code structure maps directly to the paper modules:
    1. ``TruncatedResNetEncoder`` extracts 26x26 Cartesian BEV features.
    2. ``CartesianToPolar`` resamples Cartesian features into 16x64 polar features.
    3. ``PolarMixStyle`` applies style perturbation only during training to improve cross-domain generalization.
    4. ``CircularConvBlock`` applies circular padding along the angular axis to avoid a 0/2pi boundary break.
    5. ``PolarSelfAttention`` models long-range structure over polar tokens.
    6. ``AngularGeMPool`` aggregates along the angular axis while preserving radial structure.
    7. ``RadialMixVPRHead`` outputs the final 4096-D L2-normalized descriptor.
    """

    def __init__(self, model_cfg: Dict) -> None:
        super().__init__()
        frontend_cfg = model_cfg.get("frontend", {})
        if str(frontend_cfg.get("mode", "polar")).lower() != "polar":
            raise ValueError("USVLoc only keeps the final polar frontend implementation.")

        self.backbone = TruncatedResNetEncoder(
            pretrained=bool(model_cfg.get("frontend_pretrained", False)),
            backbone_name=str(model_cfg.get("frontend_backbone_name", "resnet34")),
        )
        polar_cfg = frontend_cfg.get("polar", {})
        circular_cfg = frontend_cfg.get("circular_conv", {})
        polar_attention_cfg = frontend_cfg.get("polar_attention", {})
        aggregator_cfg = frontend_cfg.get("aggregator", {})
        theta_pool = str(frontend_cfg.get("theta_pool", "gem")).lower()
        if theta_pool != "gem":
            raise ValueError("USVLoc final release only keeps AngularGeM pooling.")

        self.cartesian_to_polar = CartesianToPolar(
            radial_bins=int(polar_cfg.get("radial_bins", 16)),
            angular_bins=int(polar_cfg.get("angular_bins", 64)),
            center_xy=polar_cfg.get("center_xy", None),
            radius_max=float(polar_cfg.get("radius_max")) if polar_cfg.get("radius_max", None) is not None else None,
            align_corners=bool(polar_cfg.get("align_corners", False)),
        )
        mix_style_cfg = polar_cfg.get("mix_style", {})
        self.polar_mix_style = PolarMixStyle(
            enabled=bool(mix_style_cfg.get("enabled", False)),
            p=float(mix_style_cfg.get("p", 0.5)),
            alpha=float(mix_style_cfg.get("alpha", 0.1)),
            eps=float(mix_style_cfg.get("eps", 1.0e-6)),
        )
        self.circular_blocks = nn.Sequential(
            *[
                CircularConvBlock(
                    channels=self.backbone.local_feat_dim,
                    kernel_size=int(circular_cfg.get("kernel_size", 3)),
                )
                for _ in range(max(int(circular_cfg.get("depth", 3)), 0))
            ]
        )
        self.polar_attention = PolarSelfAttention(
            channels=self.backbone.local_feat_dim,
            num_heads=int(polar_attention_cfg.get("num_heads", 4)),
            rope_mode=str(polar_attention_cfg.get("rope_mode", "none")),
            attn_dropout=float(polar_attention_cfg.get("attn_dropout", 0.0)),
            proj_dropout=float(polar_attention_cfg.get("proj_dropout", 0.0)),
            residual_scale_init=float(polar_attention_cfg.get("residual_scale_init", 0.1)),
            norm_eps=float(polar_attention_cfg.get("norm_eps", 1.0e-5)),
            enabled=bool(polar_attention_cfg.get("enabled", False)),
        )
        self.angular_pool = AngularGeMPool(
            p_init=float(frontend_cfg.get("theta_pool_gem_p_init", 3.0)),
            eps=float(frontend_cfg.get("theta_pool_gem_eps", 1.0e-6)),
        )
        self.global_head = RadialMixVPRHead(
            in_channels=self.backbone.local_feat_dim,
            in_rows=int(aggregator_cfg.get("in_rows", polar_cfg.get("radial_bins", 16))),
            out_channels=int(aggregator_cfg.get("out_channels", 512)),
            mix_depth=int(aggregator_cfg.get("mix_depth", 3)),
            mlp_ratio=float(aggregator_cfg.get("mlp_ratio", 1.0)),
            out_rows=int(aggregator_cfg.get("out_rows", 8)),
        )

        self.local_feat_dim = self.backbone.local_feat_dim
        self.global_descriptor_dim = int(self.global_head.output_dim)

    def _theta_pool(self, polar_features: torch.Tensor) -> torch.Tensor:
        return self.angular_pool(polar_features)

    def forward_retrieval(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return intermediate features used by retrieval and the backend.

        ``global_descriptor`` is used for place recognition;
        ``cartesian_features/local_features`` are used by the geometric backend;
        ``polar_features`` are the polar features before circular convolution and
        can be used by the polar backend or for visualization.
        """
        backbone_features = self.backbone.forward_features(x)
        # feat1 has higher resolution and is interpolated back to the input size as dense local features.
        local_features = F.interpolate(
            backbone_features["feat1"],
            size=tuple(int(v) for v in x.shape[-2:]),
            mode="bilinear",
            align_corners=False,
        )
        local_features = F.normalize(local_features, p=2, dim=1)

        # feat2 is the 26x26 Cartesian feature map from the paper and is later resampled onto the polar grid.
        cartesian_features = backbone_features["feat2"]
        polar_features = self.cartesian_to_polar(cartesian_features)
        polar_features = self.polar_mix_style(polar_features)
        backend_polar_features = polar_features
        polar_features = self.circular_blocks(polar_features)
        polar_features = self.polar_attention(polar_features)
        radial_features = self._theta_pool(polar_features)
        global_descriptor = self.global_head(radial_features)
        return {
            "global_descriptor": global_descriptor,
            "cartesian_features": cartesian_features,
            "local_features": local_features,
            "polar_features": backend_polar_features,
            "attended_polar_features": polar_features,
            "radial_features": radial_features,
        }

    def forward(self, x: torch.Tensor, return_dict: bool = False):
        output = self.forward_retrieval(x)
        if return_dict:
            return output
        return output["global_descriptor"]

    def get_global_descriptor(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_retrieval(x)["global_descriptor"]
