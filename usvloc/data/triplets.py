from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import cv2
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset

from .common import (
    ProcessedSequenceDataset,
    pose_translation_from_raw,
    resize_and_to_tensor,
)

_EXACT_DISTANCE_POOL_MAX_SAMPLES = 5000


def _sample_rotation_angle_deg() -> int:
    return int(np.random.randint(0, 360))


def _rotate_with_angle(image: np.ndarray, angle_deg: int) -> np.ndarray:
    center = (image.shape[1] // 2, image.shape[0] // 2)
    matrix = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _random_rotate(image: np.ndarray) -> np.ndarray:
    return _rotate_with_angle(image, _sample_rotation_angle_deg())


def _extract_positions(samples, dataset_name: str) -> np.ndarray:
    dataset_name = str(dataset_name).lower()
    if dataset_name == "kitti" and "pose_raw" in samples.columns:
        return np.asarray(
            [pose_translation_from_raw(pose_raw) for pose_raw in samples["pose_raw"].tolist()],
            dtype=np.float64,
        )
    if {"x_m", "y_m", "z_m"}.issubset(samples.columns):
        return samples[["x_m", "y_m", "z_m"]].to_numpy(dtype=np.float64)
    xy = samples[["x_m", "y_m"]].to_numpy(dtype=np.float64)
    zeros = np.zeros((xy.shape[0], 1), dtype=np.float64)
    return np.concatenate([xy, zeros], axis=1)


class SBEVLocFrameTripletDataset(Dataset):
    """Triplet dataset for training.

    Each sample contains a query, a positive, and multiple negative candidates.
    Positives are selected with ``positive_distance_threshold_m``. Negatives are
    sampled after nearby frames are excluded with ``negative_distance_threshold_m``.
    When hard mining is enabled, descriptors for the full training set are cached
    before each epoch, then negatives are sampled from harder pools.
    """

    def __init__(
        self,
        sequence_dir: str | Path,
        image_size: int = 201,
        split_tags: Sequence[str] | None = ("train_db",),
        max_frame_id: int | None = 2999,
        positive_distance_threshold_m: float = 5.0,
        negative_distance_threshold_m: float = 7.0,
        num_negatives: int = 10,
        seed: int = 1024,
        augment_random_rotation: bool = True,
        hard_mining_enabled: bool = True,
        hard_negative_candidate_pool_size: int = 10,
        hard_positive_mining_enabled: bool = False,
        processed_dataset_kwargs: Dict[str, object] | None = None,
    ) -> None:
        self.sequence_dir = Path(sequence_dir)
        self.base_dataset = ProcessedSequenceDataset(
            self.sequence_dir,
            image_size=image_size,
            split_tags=None,
            **(processed_dataset_kwargs or {}),
        )
        selected = self.base_dataset.frames.copy()
        if split_tags:
            selected = selected[selected["split_tag"].isin({str(tag) for tag in split_tags})]
        if max_frame_id is not None and "frame_id" in selected.columns:
            selected = selected[selected["frame_id"] <= int(max_frame_id)]
        selected = selected.reset_index().rename(columns={"index": "base_index"})
        if len(selected) == 0:
            raise RuntimeError(f"No training frames matched split_tags={list(split_tags) if split_tags else None} in {self.sequence_dir}")

        self.samples = selected
        self.base_indices = self.samples["base_index"].to_numpy(dtype=np.int64)
        self.frame_ids = self.samples["frame_id"].to_numpy(dtype=np.int64)
        self.positions = _extract_positions(self.samples, dataset_name=self.base_dataset.meta.dataset_name)
        self.image_size = int(image_size)
        self.positive_distance_threshold_m = float(positive_distance_threshold_m)
        self.negative_distance_threshold_m = float(negative_distance_threshold_m)
        if self.negative_distance_threshold_m <= self.positive_distance_threshold_m:
            raise ValueError(
                "negative_distance_threshold_m must be greater than positive_distance_threshold_m: "
                f"{self.negative_distance_threshold_m} <= {self.positive_distance_threshold_m}"
            )
        self.num_negatives = int(num_negatives)
        self.random = np.random.RandomState(int(seed))
        self.augment_random_rotation = bool(augment_random_rotation)
        self.hard_mining_enabled = bool(hard_mining_enabled)
        self.hard_negative_candidate_pool_size = max(int(hard_negative_candidate_pool_size), self.num_negatives)
        self.hard_positive_mining_enabled = bool(hard_positive_mining_enabled)
        self.num_samples = int(self.positions.shape[0])
        self.valid_local_indices, self.positive_pools, self.negative_pools = self._precompute_triplet_pools()
        self.epoch_descriptor_cache: np.ndarray | None = None
        self.cache_build_mode = False

    def _precompute_triplet_pools(self) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Precompute positive and negative pools for each anchor.

        Small sequences build the full distance matrix directly. Large
        sequences use a KD-tree to find neighbors, and negatives are sampled from
        the complement of the neighbor set to avoid O(N^2) memory growth.
        """
        n_samples = int(self.positions.shape[0])
        positive_pools: list[np.ndarray] = [np.zeros((0,), dtype=np.int64) for _ in range(n_samples)]
        negative_pools: list[np.ndarray] = [np.zeros((0,), dtype=np.int64) for _ in range(n_samples)]
        valid_local_indices: list[int] = []

        if n_samples <= _EXACT_DISTANCE_POOL_MAX_SAMPLES:
            self.negative_pool_mode = "explicit"
            delta = self.positions[:, None, :] - self.positions[None, :, :]
            dist_m = np.linalg.norm(delta, axis=-1)
            for anchor_idx in range(n_samples):
                order = np.argsort(dist_m[anchor_idx])
                ordered_dist = dist_m[anchor_idx][order]
                positive = order[(ordered_dist > 0.0) & (ordered_dist < self.positive_distance_threshold_m)]
                negative = order[ordered_dist > self.negative_distance_threshold_m]
                if positive.size == 0 or negative.size == 0:
                    continue
                positive_pools[anchor_idx] = positive.astype(np.int64, copy=False)
                negative_pools[anchor_idx] = negative.astype(np.int64, copy=False)
                valid_local_indices.append(anchor_idx)
        else:
            self.negative_pool_mode = "complement"
            neighbors = NearestNeighbors(radius=self.negative_distance_threshold_m, algorithm="kd_tree")
            neighbors.fit(self.positions)
            dist_list, ind_list = neighbors.radius_neighbors(
                self.positions,
                radius=self.negative_distance_threshold_m,
                return_distance=True,
                sort_results=True,
            )
            for anchor_idx in range(n_samples):
                distances = np.asarray(dist_list[anchor_idx], dtype=np.float64)
                indices = np.asarray(ind_list[anchor_idx], dtype=np.int64)
                positive = indices[(distances > 0.0) & (distances < self.positive_distance_threshold_m)]
                excluded = indices.astype(np.int64, copy=False)
                negative_count = n_samples - int(excluded.shape[0])
                if positive.size == 0 or negative_count <= 0:
                    continue
                positive_pools[anchor_idx] = positive.astype(np.int64, copy=False)
                negative_pools[anchor_idx] = excluded
                valid_local_indices.append(anchor_idx)

        if not valid_local_indices:
            raise RuntimeError(
                "No valid full-image triplets found. "
                f"Check thresholds pos<{self.positive_distance_threshold_m}m and neg>{self.negative_distance_threshold_m}m."
            )
        return np.asarray(valid_local_indices, dtype=np.int64), positive_pools, negative_pools

    def __len__(self) -> int:
        return int(self.valid_local_indices.shape[0])

    def _select_positive_local(self, chosen_local: int, positive_pool: np.ndarray) -> int:
        if positive_pool.size == 0:
            raise RuntimeError(f"Invalid anchor without positives at local index {chosen_local}")
        if not self.hard_mining_enabled:
            return int(positive_pool[0])
        if self.epoch_descriptor_cache is None or not self.hard_positive_mining_enabled:
            return int(positive_pool[self.random.randint(0, positive_pool.shape[0])])

        query_desc = self.epoch_descriptor_cache[chosen_local]
        positive_desc = self.epoch_descriptor_cache[positive_pool]
        positive_dist = np.sqrt(np.sum((positive_desc - query_desc.reshape(1, -1)) ** 2, axis=1))
        hardest_positive = int(np.argmax(positive_dist))
        return int(positive_pool[hardest_positive])

    def _sample_negative_locals(self, chosen_local: int, negative_pool: np.ndarray, sample_size: int) -> np.ndarray:
        sample_size = int(sample_size)
        if sample_size <= 0:
            return np.zeros((0,), dtype=np.int64)
        if self.negative_pool_mode == "explicit":
            replace = negative_pool.shape[0] < sample_size
            return self.random.choice(negative_pool, size=sample_size, replace=replace).astype(np.int64)

        excluded = np.asarray(negative_pool, dtype=np.int64)
        if excluded.shape[0] >= self.num_samples:
            raise RuntimeError(f"No valid negatives left for local index {chosen_local}")
        results: list[int] = []
        while len(results) < sample_size:
            draw_size = max((sample_size - len(results)) * 4, sample_size)
            candidates = self.random.randint(0, self.num_samples, size=draw_size, dtype=np.int64)
            keep = candidates[~np.isin(candidates, excluded)]
            if keep.size == 0:
                continue
            results.extend(int(value) for value in keep.tolist())
        return np.asarray(results[:sample_size], dtype=np.int64)

    def _sample_plan(self, index: int, candidate_pool_size: int) -> dict[str, object]:
        """Decide which frames to sample without reading images.

        This lets hard mining and normal sampling share the same logic. Actual
        image loading happens in ``_materialize_plan``.
        """
        chosen_local = int(self.valid_local_indices[int(index)])
        positive_pool = self.positive_pools[chosen_local]
        negative_pool = self.negative_pools[chosen_local]
        if positive_pool.size == 0 or negative_pool.size == 0:
            raise RuntimeError(f"Invalid anchor without positives or negatives at local index {chosen_local}")

        if self.cache_build_mode:
            positive_local = int(positive_pool[0])
            negative_locals = self._sample_negative_locals(chosen_local, negative_pool, self.num_negatives)
            return {
                "chosen_local": chosen_local,
                "positive_local": positive_local,
                "negative_locals": negative_locals,
            }

        positive_local = self._select_positive_local(chosen_local=chosen_local, positive_pool=positive_pool)
        if self.epoch_descriptor_cache is not None and self.hard_mining_enabled and self.negative_pool_mode == "explicit":
            query_desc = self.epoch_descriptor_cache[chosen_local]
            negative_desc = self.epoch_descriptor_cache[negative_pool]
            negative_dist = np.sqrt(np.sum((negative_desc - query_desc.reshape(1, -1)) ** 2, axis=1))
            pool_size = min(int(candidate_pool_size), int(negative_dist.shape[0]))
            top_pool = np.argsort(negative_dist)[:pool_size]
            sample_size = min(self.num_negatives, int(top_pool.shape[0]))
            hard_order = self.random.choice(top_pool, size=sample_size, replace=False)
            negative_locals = np.asarray(negative_pool[hard_order], dtype=np.int64)
        else:
            negative_locals = self._sample_negative_locals(chosen_local, negative_pool, int(candidate_pool_size))

        return {
            "chosen_local": chosen_local,
            "positive_local": positive_local,
            "negative_locals": negative_locals,
        }

    def _load_tensor_from_local(self, local_idx: int, apply_augmentation: bool = True) -> torch.Tensor:
        base_index = int(self.base_indices[int(local_idx)])
        image = self.base_dataset.read_raw_image(base_index)
        if apply_augmentation and self.augment_random_rotation:
            image = _random_rotate(image)
        return resize_and_to_tensor(
            image,
            self.image_size,
            input_channels=self.base_dataset.input_channels,
            normalization_divisor=self.base_dataset.normalization_divisor,
        )

    def _materialize_plan(self, plan: dict[str, object]) -> Dict[str, object]:
        chosen_local = int(plan["chosen_local"])
        positive_local = int(plan["positive_local"])
        negative_locals = np.asarray(plan["negative_locals"], dtype=np.int64)

        query = self._load_tensor_from_local(chosen_local)
        positive = self._load_tensor_from_local(positive_local)
        negatives = torch.stack([self._load_tensor_from_local(int(local_idx)) for local_idx in negative_locals], dim=0)
        return {
            "query": query,
            "positive": positive,
            "negative_candidates": negatives,
            "frame_id": int(self.frame_ids[chosen_local]),
            "dataset_index": int(chosen_local),
            "num_keypoints": 0,
        }

    def prepare_epoch_hard_mining(self, model, device: torch.device, chunk_size: int = 16, batch_size: int = 16) -> None:
        """Build the descriptor cache for hard negative mining in the current epoch."""
        if not self.hard_mining_enabled:
            self.epoch_descriptor_cache = None
            return

        was_training = model.training
        model.eval()
        descriptor_dim = int(model.global_descriptor_dim)
        descriptor_cache = np.zeros((len(self.samples), descriptor_dim), dtype=np.float32)
        for start in range(0, len(self.samples), int(batch_size)):
            end = min(start + int(batch_size), len(self.samples))
            images = torch.stack(
                [self._load_tensor_from_local(local_idx, apply_augmentation=True) for local_idx in range(start, end)],
                dim=0,
            ).to(device)
            outputs = []
            with torch.no_grad():
                for offset in range(0, int(images.shape[0]), int(chunk_size)):
                    outputs.append(model.forward_retrieval(images[offset : offset + int(chunk_size)])["global_descriptor"].detach())
            descriptor_cache[start:end] = torch.cat(outputs, dim=0).cpu().numpy().astype(np.float32, copy=False)

        if was_training:
            model.train()
        self.epoch_descriptor_cache = descriptor_cache

    def set_cache_build_mode(self, enabled: bool) -> None:
        self.cache_build_mode = bool(enabled)

    def clear_epoch_hard_mining(self) -> None:
        self.epoch_descriptor_cache = None
        self.cache_build_mode = False

    def __getitem__(self, index: int) -> Dict[str, object]:
        candidate_pool_size = self.num_negatives if self.cache_build_mode else (
            self.hard_negative_candidate_pool_size if self.hard_mining_enabled else self.num_negatives
        )
        return self._materialize_plan(self._sample_plan(int(index), candidate_pool_size))


def sbevloc_collate_fn(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Collate multiple triplet samples into one batch.

    Negative candidates are concatenated along the batch dimension and reshaped
    back to [B, num_negatives, D] in the training loop.
    """
    query = torch.stack([item["query"] for item in batch], dim=0)
    positive = torch.stack([item["positive"] for item in batch], dim=0)
    negative_candidates = torch.cat([item["negative_candidates"] for item in batch], dim=0)
    return {
        "query": query,
        "positive": positive,
        "negative_candidates": negative_candidates,
        "negative_candidates_per_query": [int(item["negative_candidates"].shape[0]) for item in batch],
        "frame_id": [item["frame_id"] for item in batch],
        "dataset_index": [int(item["dataset_index"]) for item in batch],
        "num_keypoints": [item["num_keypoints"] for item in batch],
    }
