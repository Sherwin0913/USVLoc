from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models


def _build_resnet(backbone_name: str = "resnet34", pretrained: bool = True) -> nn.Module:
    builder = getattr(tv_models, backbone_name)
    try:
        if backbone_name == "resnet34":
            weights = tv_models.ResNet34_Weights.DEFAULT if pretrained else None
            return builder(weights=weights)
        return builder(weights="DEFAULT" if pretrained else None)
    except Exception:
        return builder(pretrained=pretrained)


class TruncatedResNetEncoder(nn.Module):
    """Truncated ResNet34 backbone.

    The encoder keeps layers through layer2 and outputs 128 channels. This
    preserves enough local geometry while avoiding the cost of a full ResNet.
    ``feat1`` is used for local matching, and ``feat2`` feeds the global
    descriptor branch.
    """

    def __init__(self, pretrained: bool = True, backbone_name: str = "resnet34") -> None:
        super().__init__()
        encoder = _build_resnet(backbone_name=backbone_name, pretrained=pretrained)
        self.encoder = nn.Sequential(*list(encoder.children())[:-4])
        self.local_feat_dim = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)["feat2"]

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # encoder[0:4] are conv/bn/relu/maxpool, encoder[4] is layer1, and encoder[5] is layer2.
        stem = self.encoder[3](self.encoder[2](self.encoder[1](self.encoder[0](x))))
        feat1 = self.encoder[4](stem)
        feat2 = self.encoder[5](feat1)
        return {"feat1": feat1, "feat2": feat2}
