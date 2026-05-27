from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import yaml


@dataclass
class SequenceMeta:
    """Metadata for one processed sequence.

    This corresponds to ``meta.yaml`` in each sequence directory. Training and
    evaluation use these fields to verify that BEV resolution, input size, and
    normalization match the configuration, preventing BEV parameters from
    different datasets from being mixed.
    """

    dataset_name: str
    sequence_name: str
    pose_adapter: str
    x_range_m: Tuple[float, float]
    y_range_m: Tuple[float, float]
    meters_per_pixel: float
    source_image_height: int
    source_image_width: int
    model_input_size: int = 256
    input_channels: int = 1
    normalization_divisor: float = 255.0
    notes: str = ""


def resolve_dataset_root(processed_root: str | Path, dataset_name: str) -> Path:
    processed_root = Path(processed_root)
    dataset_name = str(dataset_name)
    candidates = [
        processed_root / dataset_name,
        processed_root / dataset_name.upper(),
        processed_root / dataset_name.lower(),
        processed_root / dataset_name.capitalize(),
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.resolve() if candidate.is_symlink() else candidate
        if normalized in seen:
            continue
        seen.add(normalized)
        if normalized.is_dir():
            return normalized
    raise RuntimeError(
        f"Dataset root not found under {processed_root} for dataset_name={dataset_name}. "
        f"Tried: {[str(path) for path in candidates]}"
    )


def resolve_sequence_dir(processed_root: str | Path, dataset_name: str, sequence_name: str | int) -> Path:
    return resolve_dataset_root(processed_root, dataset_name) / str(sequence_name)


def load_sequence(sequence_dir: str | Path) -> tuple[pd.DataFrame, SequenceMeta]:
    sequence_dir = Path(sequence_dir)
    frames = pd.read_csv(sequence_dir / "frames.csv")
    with (sequence_dir / "meta.yaml").open("r", encoding="utf-8") as handle:
        meta_raw = yaml.safe_load(handle)
    meta = SequenceMeta(**meta_raw)
    if meta.dataset_name == "kitti" and "frame_id" in frames.columns:
        frames = frames.sort_values("frame_id").reset_index(drop=True)
    return frames, meta


def validate_sequence_meta(
    meta: SequenceMeta,
    sequence_dir: str | Path,
    expected_dataset_name: str | None = None,
    expected_meters_per_pixel: float | None = None,
    expected_model_input_size: int | None = None,
) -> None:
    sequence_dir = Path(sequence_dir)
    if expected_dataset_name is not None and str(meta.dataset_name).lower() != str(expected_dataset_name).lower():
        raise RuntimeError(
            f"Dataset mismatch for {sequence_dir}: meta.dataset_name={meta.dataset_name} "
            f"expected={expected_dataset_name}"
        )
    if expected_meters_per_pixel is not None and abs(float(meta.meters_per_pixel) - float(expected_meters_per_pixel)) > 1.0e-6:
        raise RuntimeError(
            f"BEV meters_per_pixel mismatch for {sequence_dir}: meta.yaml={meta.meters_per_pixel} "
            f"expected={expected_meters_per_pixel}"
        )
    if expected_model_input_size is not None and int(meta.model_input_size) != int(expected_model_input_size):
        raise RuntimeError(
            f"BEV model_input_size mismatch for {sequence_dir}: meta.yaml={meta.model_input_size} "
            f"expected={expected_model_input_size}"
        )


def _normalize_split_filter(split_tags: Optional[Sequence[str]]) -> Optional[set[str]]:
    if split_tags is None:
        return None
    return {str(tag) for tag in split_tags}


def parse_pose_raw(pose_raw: str) -> np.ndarray:
    values = np.fromstring(str(pose_raw), sep=" ", dtype=np.float64)
    if values.size != 12:
        raise ValueError(f"Expected 12 pose values, got {values.size}")
    return values


def pose_translation_from_raw(pose_raw: str) -> tuple[float, float, float]:
    pose = parse_pose_raw(pose_raw)
    return float(pose[3]), float(pose[7]), float(pose[11])


def yaw_from_pose_raw(pose_raw: str) -> float:
    pose = parse_pose_raw(pose_raw)
    return float(np.arctan2(pose[8], pose[0]))


def resolve_image_format(meta: SequenceMeta) -> tuple[int, float]:
    input_channels = int(getattr(meta, "input_channels", 1))
    normalization_divisor = float(getattr(meta, "normalization_divisor", 255.0))
    if meta.dataset_name == "kitti":
        if input_channels == 1:
            return 3, 256.0
        return input_channels, 256.0
    return input_channels, normalization_divisor


def read_bev_image(image_path: str | Path, input_channels: int) -> np.ndarray:
    image_path = Path(image_path)
    read_flag = cv2.IMREAD_COLOR if int(input_channels) == 3 else cv2.IMREAD_GRAYSCALE
    image = cv2.imread(str(image_path), read_flag)
    if image is None:
        raise FileNotFoundError(f"Failed to read BEV image: {image_path}")
    if int(input_channels) == 3 and image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def read_kitti_train_bgr_image(image_path: str | Path, input_channels: int) -> np.ndarray:
    image_path = Path(image_path)
    read_flag = cv2.IMREAD_COLOR if int(input_channels) == 3 else cv2.IMREAD_GRAYSCALE
    image = cv2.imread(str(image_path), read_flag)
    if image is None:
        raise FileNotFoundError(f"Failed to read KITTI BEV image: {image_path}")
    if int(input_channels) == 3 and image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def read_kitti_eval_gray3_image(image_path: str | Path, input_channels: int) -> np.ndarray:
    image_path = Path(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Failed to read KITTI BEV image: {image_path}")
    if int(input_channels) == 1:
        return image
    if int(input_channels) == 3:
        return np.repeat(image[:, :, None], 3, axis=2)
    raise ValueError(f"Unsupported KITTI input_channels={input_channels}")


def resize_and_to_tensor(
    image: np.ndarray,
    image_size: int,
    input_channels: int = 1,
    normalization_divisor: float = 255.0,
) -> torch.Tensor:
    if int(input_channels) == 1:
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if int(image.shape[0]) != int(image_size) or int(image.shape[1]) != int(image_size):
            image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / float(normalization_divisor)
        return torch.from_numpy(image[None, ...])
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if int(image.shape[0]) != int(image_size) or int(image.shape[1]) != int(image_size):
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / float(normalization_divisor)
    return torch.from_numpy(np.transpose(image, (2, 0, 1)))


class ProcessedSequenceDataset(Dataset):
    """Read a processed BEV sequence.

    Each sequence directory must contain ``frames.csv``, ``meta.yaml``, and
    ``bev/*.png``. This class loads images as tensors and returns pose fields
    used for retrieval positives and backend pose-error computation.
    """

    def __init__(
        self,
        sequence_dir: str | Path,
        image_size: int = 256,
        split_tags: Optional[Sequence[str]] = None,
        kitti_original_like_loader: bool | None = None,
        kitti_loader_mode: str | None = None,
        expected_meters_per_pixel: float | None = None,
        expected_model_input_size: int | None = None,
    ) -> None:
        self.sequence_dir = Path(sequence_dir)
        self.frames, self.meta = load_sequence(self.sequence_dir)
        validate_sequence_meta(
            self.meta,
            self.sequence_dir,
            expected_meters_per_pixel=expected_meters_per_pixel,
            expected_model_input_size=expected_model_input_size,
        )
        split_filter = _normalize_split_filter(split_tags)
        if split_filter is not None:
            self.frames = self.frames[self.frames["split_tag"].isin(split_filter)].reset_index(drop=True)
        self.image_size = int(image_size)
        self.input_channels, self.normalization_divisor = resolve_image_format(self.meta)
        self.kitti_loader_mode = "native"
        if self.meta.dataset_name == "kitti":
            # KITTI training reads BGR, while evaluation reads grayscale and repeats it into 3-channel gray3.
            if kitti_loader_mode is not None:
                normalized_mode = str(kitti_loader_mode).strip().lower()
            elif kitti_original_like_loader is None:
                normalized_mode = "kitti_eval_gray3"
            elif bool(kitti_original_like_loader):
                normalized_mode = "kitti_eval_gray3"
            else:
                normalized_mode = "native"

            aliases = {
                "native": "native",
                "processed": "native",
                "default": "native",
                "kitti_train": "kitti_train_bgr",
                "kitti_train_bgr": "kitti_train_bgr",
                "train": "kitti_train_bgr",
                "kitti_eval": "kitti_eval_gray3",
                "kitti_eval_gray3": "kitti_eval_gray3",
                "eval": "kitti_eval_gray3",
                # Backward-compatible aliases for old local experiment configs.
                "bevplace2_train": "kitti_train_bgr",
                "bevplace2_train_bgr": "kitti_train_bgr",
                "bevplace2_eval": "kitti_eval_gray3",
                "bevplace2_eval_gray3": "kitti_eval_gray3",
            }
            if normalized_mode not in aliases:
                raise ValueError(
                    f"Unsupported kitti_loader_mode={kitti_loader_mode!r}. "
                    f"Expected one of {sorted(aliases.keys())}."
                )
            self.kitti_loader_mode = aliases[normalized_mode]

    def __len__(self) -> int:
        return len(self.frames)

    def image_path_for_index(self, index: int) -> Path:
        """Resolve the BEV image path from the index in frames.csv."""
        row = self.frames.iloc[int(index)]
        if self.meta.dataset_name == "kitti" and "frame_id" in row:
            frame_id = int(row["frame_id"])
            direct_path = self.sequence_dir / "bev" / f"{frame_id:06d}.png"
            if direct_path.is_file():
                return direct_path
        return self.sequence_dir / str(row["bev_path"])

    def read_raw_image(self, index: int) -> np.ndarray:
        """Read the raw BEV image before resizing and normalization."""
        image_path = self.image_path_for_index(index)
        if self.meta.dataset_name == "kitti":
            if self.kitti_loader_mode == "kitti_train_bgr":
                return read_kitti_train_bgr_image(image_path, self.input_channels)
            if self.kitti_loader_mode == "kitti_eval_gray3":
                return read_kitti_eval_gray3_image(image_path, self.input_channels)
        return read_bev_image(image_path, self.input_channels)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.frames.iloc[int(index)].to_dict()
        image = self.read_raw_image(int(index))
        x_m = float(row["x_m"])
        y_m = float(row["y_m"])
        z_m = float(row["z_m"]) if "z_m" in row and row["z_m"] == row["z_m"] else 0.0
        yaw_rad = float(row["yaw_rad"])
        if self.meta.dataset_name == "kitti" and row.get("pose_raw", None) is not None and row["pose_raw"] == row["pose_raw"]:
            try:
                x_m, y_m, z_m = pose_translation_from_raw(row["pose_raw"])
                yaw_rad = yaw_from_pose_raw(row["pose_raw"])
            except Exception:
                pass
        return {
            "image": resize_and_to_tensor(
                image,
                self.image_size,
                input_channels=self.input_channels,
                normalization_divisor=self.normalization_divisor,
            ),
            "index": int(index),
            "frame_id": int(row["frame_id"]),
            "timestamp": row["timestamp"],
            "x_m": x_m,
            "y_m": y_m,
            "z_m": z_m,
            "yaw_rad": yaw_rad,
            "split_tag": row["split_tag"],
        }
