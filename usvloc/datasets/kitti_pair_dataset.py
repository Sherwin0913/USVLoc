"""KITTI BEV pair dataset for training the external local descriptor head."""

from __future__ import annotations

import math
from typing import Callable, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def enumerate_pairs(
    poses_xz: np.ndarray,
    dist_min: float = 3.0,
    dist_max: float = 15.0,
    max_per_query: int = 4,
    max_frame_gap: int = 300,
) -> List[Tuple[int, int]]:
    poses_xz = np.asarray(poses_xz, dtype=np.float64)
    num_frames = int(poses_xz.shape[0])
    pairs: set[tuple[int, int]] = set()
    for i in range(num_frames):
        lo = max(0, i - int(max_frame_gap))
        hi = min(num_frames, i + int(max_frame_gap) + 1)
        candidates: list[tuple[float, int]] = []
        for j in range(lo, hi):
            if j == i:
                continue
            distance = float(np.linalg.norm(poses_xz[i] - poses_xz[j]))
            if float(dist_min) <= distance <= float(dist_max):
                candidates.append((distance, j))
        candidates.sort(key=lambda item: item[0])
        for _, j in candidates[: int(max_per_query)]:
            a, b = (i, j) if i < j else (j, i)
            pairs.add((a, b))
    return sorted(pairs)


def relative_pose_2d(pose_q: np.ndarray, pose_d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the 2D transform from query-local coordinates to db-local coordinates.

    ``pose`` is ``[x, z, yaw_rad]`` in the same BEV ground-plane convention used
    by the backend evaluator.
    """

    x_q, z_q, yaw_q = [float(v) for v in pose_q]
    x_d, z_d, yaw_d = [float(v) for v in pose_d]
    dyaw = yaw_q - yaw_d
    c, s = math.cos(dyaw), math.sin(dyaw)
    rotation_q_to_d = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    rotation_world_to_d = np.asarray(
        [
            [math.cos(yaw_d), math.sin(yaw_d)],
            [-math.sin(yaw_d), math.cos(yaw_d)],
        ],
        dtype=np.float64,
    )
    translation_q_to_d = rotation_world_to_d @ np.asarray([x_q - x_d, z_q - z_d], dtype=np.float64)
    return rotation_q_to_d, translation_q_to_d


class KITTIPairDataset(Dataset):
    """Return paired KITTI BEV images and their query-to-database 2D transform.

    ``bev_loader(seq, idx)`` must return ``torch.Tensor [3,201,201]``.
    ``pose_loader(seq)`` must return ``np.ndarray [N,3]`` as ``[x,z,yaw_rad]``.
    """

    def __init__(
        self,
        bev_loader: Callable[[str, int], torch.Tensor],
        pose_loader: Callable[[str], np.ndarray],
        seqs: Tuple[str, ...] = ("00", "02", "05", "06", "08"),
        dist_min: float = 3.0,
        dist_max: float = 15.0,
        max_per_query: int = 4,
        max_frame_gap: int = 300,
    ) -> None:
        self.bev_loader = bev_loader
        self.samples: list[tuple[str, int, int, np.ndarray, np.ndarray]] = []
        for seq in seqs:
            poses = np.asarray(pose_loader(str(seq)), dtype=np.float64)
            if poses.ndim != 2 or poses.shape[1] != 3:
                raise ValueError(f"pose_loader must return [N,3] [x,z,yaw_rad], got {poses.shape} for seq={seq}")
            pairs = enumerate_pairs(
                poses[:, :2],
                dist_min=float(dist_min),
                dist_max=float(dist_max),
                max_per_query=int(max_per_query),
                max_frame_gap=int(max_frame_gap),
            )
            for i, j in pairs:
                self.samples.append((str(seq), int(i), int(j), poses[i].copy(), poses[j].copy()))
        print(f"[KITTIPairDataset] {len(self.samples)} pairs from seqs {seqs}", flush=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq, i, j, pose_q, pose_d = self.samples[int(idx)]
        bev_q = self.bev_loader(seq, i)
        bev_d = self.bev_loader(seq, j)
        rotation, translation = relative_pose_2d(pose_q, pose_d)

        if np.random.rand() < 0.5:
            bev_q, bev_d = bev_d, bev_q
            rotation = rotation.T
            translation = -rotation @ translation

        return {
            "bev_q": bev_q.float(),
            "bev_d": bev_d.float(),
            "R_qd": torch.from_numpy(rotation.astype(np.float32)),
            "t_qd": torch.from_numpy(translation.astype(np.float32)),
        }
