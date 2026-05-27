from __future__ import annotations

from typing import Dict

import torch


def lazy_triplet_loss(
    anchor_descriptor: torch.Tensor,
    positive_descriptor: torch.Tensor,
    negative_descriptors: torch.Tensor,
    margin: float = 0.3,
) -> tuple[torch.Tensor, Dict[str, float]]:
    positive_distance = torch.norm(anchor_descriptor - positive_descriptor, dim=-1)
    negative_distances = torch.norm(anchor_descriptor.unsqueeze(1) - negative_descriptors, dim=-1)
    hardest_negative_distance = negative_distances.min(dim=1).values
    distance_gap = hardest_negative_distance - positive_distance
    loss = torch.clamp(positive_distance - hardest_negative_distance + float(margin), min=0.0).mean()
    stats = {
        "loss": float(loss.detach().cpu().item()),
        "positive_distance": float(positive_distance.detach().mean().cpu().item()),
        "hardest_negative_distance": float(hardest_negative_distance.detach().mean().cpu().item()),
        "distance_gap": float(distance_gap.detach().mean().cpu().item()),
    }
    return loss, stats
