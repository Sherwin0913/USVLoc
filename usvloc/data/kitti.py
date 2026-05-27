"""KITTI helpers used by the external local-head training script."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .common import ProcessedSequenceDataset, load_sequence, pose_translation_from_raw, resolve_sequence_dir, yaw_from_pose_raw


DEFAULT_PROCESSED_ROOT = Path("data")
DEFAULT_IMAGE_SIZE = 201
DEFAULT_KITTI_LOADER_MODE = "kitti_eval_gray3"


@lru_cache(maxsize=16)
def _cached_dataset(
    sequence: str,
    processed_root: str = str(DEFAULT_PROCESSED_ROOT),
    image_size: int = DEFAULT_IMAGE_SIZE,
    kitti_loader_mode: str = DEFAULT_KITTI_LOADER_MODE,
) -> ProcessedSequenceDataset:
    return ProcessedSequenceDataset(
        resolve_sequence_dir(processed_root, "kitti", str(sequence)),
        image_size=int(image_size),
        split_tags=None,
        kitti_loader_mode=str(kitti_loader_mode),
    )


@lru_cache(maxsize=16)
def load_kitti_poses_xz_yaw(
    sequence: str,
    processed_root: str = str(DEFAULT_PROCESSED_ROOT),
) -> np.ndarray:
    sequence_dir = resolve_sequence_dir(processed_root, "kitti", str(sequence))
    frames, _ = load_sequence(sequence_dir)
    poses = np.zeros((len(frames), 3), dtype=np.float64)
    for row_idx, row in frames.iterrows():
        if "pose_raw" in row and row["pose_raw"] == row["pose_raw"]:
            x_m, _vertical_y_m, z_m = pose_translation_from_raw(str(row["pose_raw"]))
            yaw_rad = yaw_from_pose_raw(str(row["pose_raw"]))
        else:
            x_m = float(row["x_m"])
            z_m = float(row["z_m"]) if "z_m" in row and row["z_m"] == row["z_m"] else float(row["y_m"])
            yaw_rad = float(row["yaw_rad"])
        poses[int(row_idx)] = (float(x_m), float(z_m), float(yaw_rad))
    return poses


def load_kitti_bev(
    sequence: str,
    idx: int,
    processed_root: str = str(DEFAULT_PROCESSED_ROOT),
    image_size: int = DEFAULT_IMAGE_SIZE,
    kitti_loader_mode: str = DEFAULT_KITTI_LOADER_MODE,
) -> torch.Tensor:
    dataset = _cached_dataset(str(sequence), str(processed_root), int(image_size), str(kitti_loader_mode))
    return dataset[int(idx)]["image"]
