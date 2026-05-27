from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from usvloc.config import load_config
from usvloc.models import USVLoc

from .local_head_external import ExternalLocalHead


class NetVLAD(nn.Module):
    """Reimplementation of the NetVLAD aggregation layer used by the BEVPlace++ baseline."""

    def __init__(self, num_clusters: int = 64, dim: int = 128) -> None:
        super().__init__()
        self.num_clusters = int(num_clusters)
        self.dim = int(dim)
        self.conv = nn.Conv2d(self.dim, self.num_clusters, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(self.num_clusters, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels = x.shape[:2]
        x_flat = x.view(batch_size, channels, -1)
        soft_assign = self.conv(x).view(batch_size, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        vlad = torch.zeros(
            [batch_size, self.num_clusters, channels],
            dtype=x.dtype,
            layout=x.layout,
            device=x.device,
        )
        for cluster_idx in range(self.num_clusters):
            residual = x_flat.unsqueeze(0).permute(1, 0, 2, 3) - self.centroids[
                cluster_idx : cluster_idx + 1,
                :,
            ].expand(x_flat.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:, cluster_idx : cluster_idx + 1, :].unsqueeze(2)
            vlad[:, cluster_idx : cluster_idx + 1, :] = residual.sum(dim=-1)
        vlad = F.normalize(vlad, p=2, dim=2)
        vlad = vlad.view(x.size(0), -1)
        return F.normalize(vlad, p=2, dim=1)


def _build_resnet34_encoder() -> nn.Sequential:
    encoder = tv_models.resnet34(weights=None)
    return nn.Sequential(*list(encoder.children())[:-4])


class DeviceSafeREM(nn.Module):
    """Device-safe BEVPlace++ REM frontend.

    The original BEVPlace++ REM rotates inputs by multiple angles, encodes them,
    rotates the features back, and takes a max to obtain rotation-enhanced
    Cartesian BEV features. This keeps the same computation while avoiding the
    hard-coded device assumptions in the original repository.
    """

    def __init__(self, rotations: int = 8) -> None:
        super().__init__()
        self.encoder = _build_resnet34_encoder()
        angles = -torch.arange(0, 359.00001, 360.0 / float(rotations)) / 180.0 * torch.pi
        self.register_buffer("angles", angles, persistent=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        equivariant_features = []
        batch_size = x.size(0)
        init_size = None
        for angle in self.angles:
            aff = torch.zeros(batch_size, 2, 3, device=x.device, dtype=x.dtype)
            aff[:, 0, 0] = torch.cos(-angle)
            aff[:, 0, 1] = torch.sin(-angle)
            aff[:, 1, 0] = -torch.sin(-angle)
            aff[:, 1, 1] = torch.cos(-angle)
            grid = F.affine_grid(aff, torch.Size(x.size()), align_corners=True)
            warped = F.grid_sample(x, grid, align_corners=True, mode="bicubic")
            out = self.encoder(warped)
            if init_size is None:
                init_size = out.size()

            aff = torch.zeros(batch_size, 2, 3, device=x.device, dtype=x.dtype)
            aff[:, 0, 0] = torch.cos(angle)
            aff[:, 0, 1] = torch.sin(angle)
            aff[:, 1, 0] = -torch.sin(angle)
            aff[:, 1, 1] = torch.cos(angle)
            grid = F.affine_grid(aff, torch.Size(init_size), align_corners=True)
            out = F.grid_sample(out, grid, align_corners=True, mode="bicubic")
            equivariant_features.append(out.unsqueeze(-1))

        features = torch.cat(equivariant_features, dim=-1)
        features = torch.max(features, dim=-1, keepdim=False)[0]

        identity = torch.zeros(batch_size, 2, 3, device=x.device, dtype=x.dtype)
        identity[:, 0, 0] = 1.0
        identity[:, 1, 1] = 1.0
        b, c, h, w = x.size()
        grid = F.affine_grid(identity, torch.Size((b, c, h // 4, w // 4)), align_corners=True)
        netvlad_input = F.grid_sample(features, grid, align_corners=True, mode="bicubic")
        netvlad_input = F.normalize(netvlad_input, dim=1)

        grid = F.affine_grid(identity, torch.Size((b, c, h, w)), align_corners=True)
        local_features = F.grid_sample(features, grid, align_corners=True, mode="bicubic")
        local_features = F.normalize(local_features, dim=1)
        return netvlad_input, local_features


class DeviceSafeREIN(nn.Module):
    """Wrapper for the BEVPlace++ REM + NetVLAD retrieval network."""

    def __init__(self) -> None:
        super().__init__()
        self.rem = DeviceSafeREM(rotations=8)
        self.pooling = NetVLAD(num_clusters=64, dim=128)
        self.local_feat_dim = 128
        self.global_feat_dim = 128 * 64

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        netvlad_input, local_features = self.rem(x)
        global_descriptor = self.pooling(netvlad_input)
        return {
            "netvlad_input": netvlad_input,
            "local_features": local_features,
            "global_descriptor": global_descriptor,
        }


class USVLocAdapter:
    """USVLoc backend evaluation adapter.

    Backend evaluation needs a unified interface: global descriptors are used
    for retrieval, and local features are used for geometric verification. The
    USVLoc global descriptor comes from the polar frontend; the default local
    features come from ``cartesian_features`` because standard BEV RANSAC works
    in Cartesian pixel coordinates.
    """

    name = "usvloc"
    local_feature_source = "cartesian_features"

    def __init__(self, model: USVLoc, device: torch.device) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.external_local_head: ExternalLocalHead | None = None
        self.external_local_head_ckpt: str | None = None

    def load_external_local_head(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = Path(checkpoint_path)
        head = ExternalLocalHead(in_dim=128, out_dim=128, out_size=201).to(self.device).eval()
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        head.load_state_dict(state_dict, strict=True)
        for parameter in head.parameters():
            parameter.requires_grad_(False)
        self.external_local_head = head
        self.external_local_head_ckpt = str(checkpoint_path.resolve())
        self.local_feature_source = "external_local_head"
        print(f"[USVLocAdapter] loaded ExternalLocalHead from {self.external_local_head_ckpt}", flush=True)

    def _local_from_output(
        self,
        output: Dict[str, torch.Tensor],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        """Extract backend local features from USVLoc forward output."""
        cartesian = output["cartesian_features"]
        if self.external_local_head is not None:
            local = self.external_local_head(cartesian.to(self.device))
            if tuple(int(v) for v in local.shape[-2:]) != tuple(int(v) for v in output_size):
                local = F.interpolate(local, size=output_size, mode="bilinear", align_corners=False)
            return F.normalize(local, dim=1)

        local = cartesian
        if tuple(int(v) for v in local.shape[-2:]) != tuple(int(v) for v in output_size):
            local = F.interpolate(local, size=output_size, mode="bilinear", align_corners=False)
        return F.normalize(local, dim=1)

    def forward_global(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model.forward_retrieval(images.to(self.device))
        return output["global_descriptor"]

    def forward_retrieval_outputs(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model.forward_retrieval(images.to(self.device))

    def forward_backend_features(
        self,
        images: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.forward_retrieval_outputs(images)
        descriptors = output["global_descriptor"]
        polar = output["polar_features"]
        if output_size is None:
            output_size = tuple(int(v) for v in images.shape[-2:])
        local = self._local_from_output(output, output_size=output_size)
        return descriptors, polar, local

    def forward_pair_features(
        self,
        query_images: torch.Tensor,
        candidate_images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        images = torch.cat([query_images, candidate_images], dim=0).to(self.device)
        output = self.model.forward_retrieval(images)
        descriptors = output["global_descriptor"]
        local = self._local_from_output(output, output_size=tuple(int(v) for v in images.shape[-2:]))
        batch_size = int(query_images.shape[0])
        return (
            descriptors[:batch_size],
            descriptors[batch_size:],
            local[:batch_size],
            local[batch_size:],
        )

class BEVPlacePPAdapter:
    """Unified backend adapter for the BEVPlace++ baseline."""

    name = "bevplacepp"
    local_feature_source = "bevplacepp_rem_local_features"

    def __init__(self, model: DeviceSafeREIN, device: torch.device) -> None:
        self.model = model.to(device).eval()
        self.device = device

    def forward_global(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model(images.to(self.device))
        return output["global_descriptor"]

    def forward_pair_features(
        self,
        query_images: torch.Tensor,
        candidate_images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        images = torch.cat([query_images, candidate_images], dim=0).to(self.device)
        output = self.model(images)
        descriptors = output["global_descriptor"]
        local = output["local_features"]
        local = F.normalize(local, dim=1)
        batch_size = int(query_images.shape[0])
        return (
            descriptors[:batch_size],
            descriptors[batch_size:],
            local[:batch_size],
            local[batch_size:],
        )

    def forward_local_features(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model(images.to(self.device))
        return F.normalize(output["local_features"], dim=1)


def _load_state_dict(checkpoint_path: str | Path, map_location: torch.device | str):
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"], checkpoint
    return checkpoint, checkpoint


def load_usvloc_adapter(
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: torch.device,
    overrides: list[str] | None = None,
) -> tuple[USVLocAdapter, Dict]:
    """Load a USVLoc checkpoint and return the unified backend adapter."""
    cfg = load_config(config_path, overrides=overrides or [])
    model = USVLoc(cfg["model"])
    state_dict, checkpoint = _load_state_dict(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    adapter = USVLocAdapter(model, device=device)
    metadata = {
        "model_type": "usvloc",
        "config": str(Path(config_path).resolve()),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch", None) if isinstance(checkpoint, dict) else None,
        "local_feature_source": adapter.local_feature_source,
    }
    return adapter, metadata


def load_bevplacepp_adapter(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[BEVPlacePPAdapter, Dict]:
    """Load a BEVPlace++ checkpoint and return the unified backend adapter."""
    model = DeviceSafeREIN()
    state_dict, checkpoint = _load_state_dict(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Failed to load BEVPlace++ checkpoint: missing={missing}, unexpected={unexpected}")
    adapter = BEVPlacePPAdapter(model, device=device)
    metadata = {
        "model_type": "bevplacepp",
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch", None) if isinstance(checkpoint, dict) else None,
        "local_feature_source": adapter.local_feature_source,
    }
    return adapter, metadata


def descriptors_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32, copy=False)
