from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..data.common import resize_and_to_tensor
from ..io import ensure_dir, save_json, save_tsv
from .retrieval import compute_recall_at_k, compute_recall_at_one_percent, search_topk


USVINLAND_DEFAULT_SEQUENCES: tuple[str, ...] = (
    "H05_7_Sequence_160_270",
    "H05_9_Sequence_115_700",
    "N02_4_Sequence_155_370",
    "N03_2_Sequence_80_536",
    "N03_3_Sequence_605_760",
    "N03_4_Sequence_440_523",
    "N03_5_Sequence_12_340",
    "W06_2_Sequence_57_115",
)


@dataclass
class USVInlandSequence:
    sequence_name: str
    frame_ids: np.ndarray
    frame_paths: list[Path]
    timestamps: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    yaw_rad: np.ndarray
    nav_match_dt_s: np.ndarray


class USVInlandDataset(Dataset):
    """Reader for raw USVInland BEV images."""

    def __init__(
        self,
        sequence: USVInlandSequence,
        image_size: int,
        input_channels: int,
        normalization_divisor: float,
    ) -> None:
        self.sequence = sequence
        self.image_size = int(image_size)
        self.input_channels = int(input_channels)
        self.normalization_divisor = float(normalization_divisor)

    def __len__(self) -> int:
        return len(self.sequence.frame_paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_path = self.sequence.frame_paths[int(index)]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR if self.input_channels == 3 else cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Failed to read BEV image: {image_path}")
        if self.input_channels == 1 and image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return {
            "image": resize_and_to_tensor(
                image,
                self.image_size,
                input_channels=self.input_channels,
                normalization_divisor=self.normalization_divisor,
            ),
            "index": int(index),
        }


def discover_usvinland_sequences(raw_root: str | Path) -> tuple[str, ...]:
    raw_root = Path(raw_root)
    sequence_names: list[str] = []
    for bev_dir in sorted(raw_root.glob("*-BEV")):
        if not bev_dir.is_dir():
            continue
        sequence_name = bev_dir.name[:-4]
        if (raw_root / f"{sequence_name}-INS").is_dir() and (raw_root / f"{sequence_name}-Lidar").is_dir():
            sequence_names.append(sequence_name)
    if not sequence_names:
        raise RuntimeError(f"No valid USVInland sequences discovered under {raw_root}")
    return tuple(sequence_names)


def load_usvinland_sequence(raw_root: str | Path, sequence_name: str) -> USVInlandSequence:
    """Load one USVInland sequence and align LiDAR frames with INS/GPS poses by timestamp."""
    raw_root = Path(raw_root)
    bev_dir = raw_root / f"{sequence_name}-BEV"
    ins_dir = raw_root / f"{sequence_name}-INS"
    lidar_dir = raw_root / f"{sequence_name}-Lidar"
    if not bev_dir.is_dir():
        raise FileNotFoundError(f"USVInland BEV dir not found: {bev_dir}")
    if not ins_dir.is_dir():
        raise FileNotFoundError(f"USVInland INS dir not found: {ins_dir}")
    if not lidar_dir.is_dir():
        raise FileNotFoundError(f"USVInland lidar dir not found: {lidar_dir}")

    frame_paths = sorted(bev_dir.glob("*.png"), key=lambda path: int(path.stem))
    if not frame_paths:
        raise RuntimeError(f"No BEV frames found in {bev_dir}")
    frame_ids = np.asarray([int(path.stem) for path in frame_paths], dtype=np.int64)

    lidar_timestamps = np.loadtxt(lidar_dir / "Laser_Timestamp_Middle.txt", dtype=np.float64)
    if np.max(frame_ids) >= int(lidar_timestamps.shape[0]):
        raise RuntimeError(
            f"Frame ids exceed lidar timestamps for {sequence_name}: "
            f"max_frame_id={int(np.max(frame_ids))} timestamps={int(lidar_timestamps.shape[0])}"
        )
    frame_timestamps = lidar_timestamps[frame_ids]

    gps = pd.read_csv(ins_dir / "GPSBase.csv", header=None)
    if gps.shape[1] < 11:
        raise RuntimeError(f"Unexpected GPSBase.csv format for {sequence_name}: shape={gps.shape}")
    gps_t = gps.iloc[:, 0].to_numpy(dtype=np.float64)
    gps_x = gps.iloc[:, 1].to_numpy(dtype=np.float64)
    gps_y = gps.iloc[:, 2].to_numpy(dtype=np.float64)
    gps_heading_deg = gps.iloc[:, 10].to_numpy(dtype=np.float64)

    insert_idx = np.searchsorted(gps_t, frame_timestamps, side="left")
    idx_right = np.clip(insert_idx, 0, len(gps_t) - 1)
    idx_left = np.clip(insert_idx - 1, 0, len(gps_t) - 1)
    choose_right = np.abs(gps_t[idx_right] - frame_timestamps) < np.abs(gps_t[idx_left] - frame_timestamps)
    matched_idx = np.where(choose_right, idx_right, idx_left)

    yaw_rad = np.deg2rad(gps_heading_deg[matched_idx].astype(np.float64))
    return USVInlandSequence(
        sequence_name=sequence_name,
        frame_ids=frame_ids,
        frame_paths=frame_paths,
        timestamps=frame_timestamps.astype(np.float64),
        x_m=gps_x[matched_idx].astype(np.float64),
        y_m=gps_y[matched_idx].astype(np.float64),
        yaw_rad=yaw_rad.astype(np.float64),
        nav_match_dt_s=np.abs(gps_t[matched_idx] - frame_timestamps).astype(np.float64),
    )


def extract_usvinland_descriptors(
    model,
    sequence: USVInlandSequence,
    image_size: int,
    input_channels: int,
    normalization_divisor: float,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    include_tta: bool = False,
) -> np.ndarray:
    dataset = USVInlandDataset(
        sequence=sequence,
        image_size=image_size,
        input_channels=input_channels,
        normalization_divisor=normalization_divisor,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(max(0, num_workers)),
        pin_memory=device.type == "cuda",
    )
    descriptors: list[np.ndarray] = []
    total_batches = len(loader)
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=device.type == "cuda")
            out = model.forward_retrieval(images)
            desc = out["global_descriptor"].detach().cpu().numpy().astype(np.float32, copy=False)
            if bool(include_tta):
                per_rotation = [desc]
                for rotation_k in (1, 2, 3):
                    rotated = torch.rot90(images, k=int(rotation_k), dims=(-2, -1))
                    rotated_out = model.forward_retrieval(rotated)
                    per_rotation.append(
                        rotated_out["global_descriptor"].detach().cpu().numpy().astype(np.float32, copy=False)
                    )
                descriptors.append(np.stack(per_rotation, axis=1).astype(np.float32, copy=False))
            else:
                descriptors.append(desc)
            if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == total_batches:
                print(
                    f"[USVInland][Eval] {sequence.sequence_name} feature bank {batch_idx}/{total_batches}",
                    flush=True,
                )
    return np.concatenate(descriptors, axis=0).astype(np.float32, copy=False)


def search_topk_with_query_tta(
    db_descs: np.ndarray,
    query_descs: np.ndarray,
    topk: int,
    metric: str = "l2",
    use_gpu: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    query_descs = np.asarray(query_descs, dtype=np.float32)
    if query_descs.ndim != 3:
        return search_topk(db_descs, query_descs, topk=topk, metric=metric, use_gpu=use_gpu)
    num_queries, num_rot, dim = query_descs.shape
    flat_queries = query_descs.reshape(num_queries * num_rot, dim)
    search_k = min(max(int(topk) * 2, int(topk)), int(np.asarray(db_descs).shape[0]))
    flat_scores, flat_indices = search_topk(db_descs, flat_queries, topk=search_k, metric=metric, use_gpu=use_gpu)
    scores = flat_scores.reshape(num_queries, num_rot, search_k)
    indices = flat_indices.reshape(num_queries, num_rot, search_k)
    out_scores = np.empty((num_queries, int(topk)), dtype=np.float32)
    out_indices = np.empty((num_queries, int(topk)), dtype=np.int64)
    prefer_low = str(metric).lower() == "l2"
    for query_i in range(num_queries):
        merged: dict[int, float] = {}
        for rot_i in range(num_rot):
            for score, index in zip(scores[query_i, rot_i], indices[query_i, rot_i]):
                index = int(index)
                score = float(score)
                if index < 0:
                    continue
                if index not in merged or (score < merged[index] if prefer_low else score > merged[index]):
                    merged[index] = score
        ordered = sorted(merged.items(), key=lambda item: item[1], reverse=not prefer_low)[: int(topk)]
        while len(ordered) < int(topk):
            fill_index = int(ordered[-1][0]) if ordered else 0
            fill_score = float("inf") if prefer_low else float("-inf")
            ordered.append((fill_index, fill_score))
        out_indices[query_i] = np.asarray([index for index, _ in ordered], dtype=np.int64)
        out_scores[query_i] = np.asarray([score for _, score in ordered], dtype=np.float32)
    return out_scores, out_indices


def split_usvinland_indices(num_frames: int, split_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """Split database/query indices in temporal order.

    The default 0.5 ratio uses the first half as the database and the second
    half as queries.
    """
    split_ratio = float(split_ratio)
    if not 0.0 < split_ratio < 1.0:
        raise ValueError(f"split_ratio must be in (0, 1), got {split_ratio}")
    split_index = int(round(float(num_frames) * split_ratio))
    split_index = max(1, min(int(num_frames) - 1, split_index))
    db_indices = np.arange(0, split_index, dtype=np.int64)
    query_indices = np.arange(split_index, num_frames, dtype=np.int64)
    return db_indices, query_indices


def _write_place_outputs(output_dir: Path, per_sequence: list[dict[str, object]]) -> dict[str, object]:
    mean_r1 = float(np.mean([row["Recall@1"] for row in per_sequence])) if per_sequence else 0.0
    mean_r5 = float(np.mean([row["Recall@5"] for row in per_sequence])) if per_sequence else 0.0
    mean_r1p = float(np.mean([row["Recall@1%"] for row in per_sequence])) if per_sequence else 0.0
    payload = {
        "sequences": [str(row["sequence"]) for row in per_sequence],
        "mean_recall_at_1": mean_r1,
        "mean_recall_at_5": mean_r5,
        "mean_recall_at_1_percent": mean_r1p,
        "per_sequence": per_sequence,
        "paper": {
            "per_sequence": [{"sequence": str(row["sequence"]), "Recall@1": float(row["Recall@1"])} for row in per_sequence],
            "Mean Recall@1": mean_r1,
        },
        "diagnostics": {
            "per_sequence": per_sequence,
            "Mean Recall@1": mean_r1,
            "Mean Recall@5": mean_r5,
            "Mean Recall@1%": mean_r1p,
        },
    }
    save_json(output_dir / "paper_place_all.json", payload)
    save_tsv(output_dir / "paper_place_all.tsv", per_sequence)
    save_json(output_dir / "paper_place.json", payload["paper"])
    save_tsv(output_dir / "paper_place.tsv", payload["paper"]["per_sequence"])
    save_json(output_dir / "diagnostics_place.json", payload["diagnostics"])
    save_tsv(output_dir / "diagnostics_place.tsv", per_sequence)
    return payload


def evaluate_usvinland_place(
    model,
    cfg: dict,
    device: torch.device,
    output_dir: str | Path | None = None,
    raw_root: str | Path = "data/USVInlandRaw",
    sequences: Sequence[str] | None = None,
    positive_radius_m: float = 5.0,
    split_ratio: float = 0.5,
    eval_batch_size: int | None = None,
    num_workers: int = 4,
    normalization_divisor: float = 255.0,
    faiss_gpu: bool = False,
    query_tta: bool = False,
) -> dict[str, object]:
    """Evaluate USVInland place recognition.

    USVInland does not use ``processed_root``. It reads ``*-BEV``, ``*-INS``,
    and ``*-Lidar`` folders directly under ``raw_root``.
    """
    raw_root = Path(raw_root).resolve()
    if sequences is None:
        sequences = discover_usvinland_sequences(raw_root)
    else:
        sequences = [str(sequence) for sequence in sequences]
        if len(sequences) == 0:
            sequences = list(discover_usvinland_sequences(raw_root))
    image_size = int(cfg["model"].get("input_size", 201))
    input_channels = int(cfg["model"].get("input_channels", 3))
    batch_size = int(eval_batch_size or cfg.get("evaluation", {}).get("eval_batch_size", 64))

    per_sequence_rows: list[dict[str, object]] = []
    sequence_summaries: list[dict[str, object]] = []
    for sequence_name in sequences:
        sequence = load_usvinland_sequence(raw_root, sequence_name)
        descriptors = extract_usvinland_descriptors(
            model=model,
            sequence=sequence,
            image_size=image_size,
            input_channels=input_channels,
            normalization_divisor=float(normalization_divisor),
            batch_size=batch_size,
            num_workers=int(num_workers),
            device=device,
            include_tta=bool(query_tta),
        )
        db_indices, query_indices = split_usvinland_indices(len(sequence.frame_paths), float(split_ratio))
        db_descs = descriptors[db_indices, 0, :] if bool(query_tta) else descriptors[db_indices]
        query_descs = descriptors[query_indices] if bool(query_tta) else descriptors[query_indices]
        db_xy = np.stack([sequence.x_m[db_indices], sequence.y_m[db_indices]], axis=1).astype(np.float32, copy=False)
        query_xy = np.stack([sequence.x_m[query_indices], sequence.y_m[query_indices]], axis=1).astype(np.float32, copy=False)

        one_percent_k = max(1, int(round(len(db_descs) / 100.0)))
        topk = max(5, one_percent_k)
        _, predictions = search_topk_with_query_tta(
            db_descs,
            query_descs,
            topk=topk,
            metric="l2",
            use_gpu=bool(faiss_gpu),
        )
        recall_k = compute_recall_at_k(
            db_xy=db_xy,
            query_xy=query_xy,
            predictions=predictions,
            ks=[1, 5],
            positive_radius_m=float(positive_radius_m),
        )
        recall_one_percent = compute_recall_at_one_percent(
            db_xy=db_xy,
            query_xy=query_xy,
            predictions=predictions,
            positive_radius_m=float(positive_radius_m),
        )
        row = {
            "sequence": sequence_name,
            "Recall@1": float(recall_k["Recall@1"]),
            "Recall@5": float(recall_k["Recall@5"]),
            "Recall@1%": float(recall_one_percent),
        }
        per_sequence_rows.append(row)
        sequence_summaries.append(
            {
                "sequence": sequence_name,
                "num_frames": int(len(sequence.frame_paths)),
                "db_frames": int(len(db_indices)),
                "query_frames": int(len(query_indices)),
                "mean_nav_match_dt_s": float(np.mean(sequence.nav_match_dt_s)),
                "max_nav_match_dt_s": float(np.max(sequence.nav_match_dt_s)),
                "Recall@1": float(recall_k["Recall@1"]),
                "Recall@5": float(recall_k["Recall@5"]),
                "Recall@1%": float(recall_one_percent),
            }
        )
        print(json.dumps(sequence_summaries[-1], ensure_ascii=False), flush=True)

    place_payload = _write_place_outputs(ensure_dir(output_dir) / "place", per_sequence_rows) if output_dir is not None else {
        "sequences": [str(row["sequence"]) for row in per_sequence_rows],
        "mean_recall_at_1": float(np.mean([row["Recall@1"] for row in per_sequence_rows])) if per_sequence_rows else 0.0,
        "mean_recall_at_5": float(np.mean([row["Recall@5"] for row in per_sequence_rows])) if per_sequence_rows else 0.0,
        "mean_recall_at_1_percent": float(np.mean([row["Recall@1%"] for row in per_sequence_rows])) if per_sequence_rows else 0.0,
        "per_sequence": per_sequence_rows,
    }
    summary = {
        "protocol": "usvinland_intra_sequence_temporal_split",
        "raw_root": str(raw_root),
        "split_ratio": float(split_ratio),
        "positive_radius_m": float(positive_radius_m),
        "normalization_divisor": float(normalization_divisor),
        "faiss_gpu": bool(faiss_gpu),
        "sequence_summaries": sequence_summaries,
        "place": place_payload,
        "query_tta": int(bool(query_tta)),
        "query_tta_rotations_deg": [0, 90, 180, 270] if bool(query_tta) else [],
    }
    if output_dir is not None:
        save_json(Path(output_dir) / "summary.json", summary)
    return summary
