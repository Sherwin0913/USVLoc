from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from usvloc.data.common import ProcessedSequenceDataset
from usvloc.data.splits import build_kitti_place_spec, build_nclt_place_specs, build_pohang_place_specs, build_usvinland_place_specs
from usvloc.evaluation.retrieval import search_topk
from usvloc.io import ensure_dir, save_json, save_tsv

from .frontends import descriptors_to_numpy
from .metrics import (
    area_under_pr,
    best_f1_and_threshold,
    max_recall_at_precision,
    mean_or_zero,
    precision_recall_curve,
    summarize_runtime,
)
from .ransac import SparseRansacBackend
from .types import FeatureBank, PairResult


@dataclass
class RetrievalSpec:
    dataset_name: str
    sequence_name: str
    database_sequence_name: str
    query_sequence_dir: Path
    database_sequence_dir: Path
    query_indices: np.ndarray
    database_indices: np.ndarray
    positive_radius_m: float


class _DescriptorIndex:
    def __init__(self, db_descs: np.ndarray, metric: str = "l2", use_gpu: bool = False) -> None:
        self.db_descs = np.asarray(db_descs, dtype=np.float32)
        self.metric = str(metric).lower()
        self.use_gpu = bool(use_gpu)
        self.index = None
        try:
            import faiss  # type: ignore

            cpu_index = faiss.IndexFlatL2(self.db_descs.shape[1]) if self.metric == "l2" else faiss.IndexFlatIP(self.db_descs.shape[1])
            if self.use_gpu and hasattr(faiss, "StandardGpuResources") and torch.cuda.is_available():
                resources = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(resources, torch.cuda.current_device(), cpu_index)
            else:
                self.index = cpu_index
            self.index.add(self.db_descs)
        except Exception:
            self.index = None

    def search(self, query_descs: np.ndarray, topk: int = 1) -> tuple[np.ndarray, np.ndarray]:
        query_descs = np.asarray(query_descs, dtype=np.float32)
        if self.index is not None:
            return self.index.search(query_descs, int(topk))
        return search_topk(self.db_descs, query_descs, topk=int(topk), metric=self.metric, use_gpu=False)


def _to_numpy_position(batch: Dict[str, torch.Tensor], dataset_name: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = batch["x_m"].detach().cpu().numpy().astype(np.float32)
    y = batch["y_m"].detach().cpu().numpy().astype(np.float32)
    z = batch["z_m"].detach().cpu().numpy().astype(np.float32)
    yaw = batch["yaw_rad"].detach().cpu().numpy().astype(np.float32)
    position = np.stack([x, y, z], axis=1)
    # BEVPlace2 KITTI positives are computed in raw 3D pose space [x,y,z],
    # while relative BEV pose errors must use the ground-plane [x,z] + yaw.
    # Non-KITTI processed datasets already store their BEV plane as [x,y].
    if str(dataset_name).lower() == "kitti":
        xy = np.stack([x, z], axis=1)
    else:
        xy = np.stack([x, y], axis=1)
    return position, xy, yaw


def _extract_feature_bank(
    adapter,
    sequence_dir: str | Path,
    image_size: int,
    batch_size: int,
    device: torch.device,
    cache: Dict[str, FeatureBank] | None,
    kitti_loader_mode: str,
    num_workers: int,
    include_tta: bool = False,
) -> FeatureBank:
    key = (str(Path(sequence_dir).resolve()), bool(include_tta))
    if cache is not None and key in cache:
        return cache[key]

    dataset = ProcessedSequenceDataset(
        sequence_dir,
        image_size=int(image_size),
        split_tags=None,
        kitti_loader_mode=str(kitti_loader_mode),
    )
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers))
    descriptors: List[np.ndarray] = []
    tta_descriptors: List[np.ndarray] = []
    positions: List[np.ndarray] = []
    xys: List[np.ndarray] = []
    yaws: List[np.ndarray] = []
    adapter.model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            desc = adapter.forward_global(images)
            descriptors.append(descriptors_to_numpy(desc))
            if bool(include_tta):
                if hasattr(adapter, "forward_global_tta"):
                    tta = adapter.forward_global_tta(images)
                else:
                    per_rotation = [desc]
                    for rotation_k in (1, 2, 3):
                        rotated = torch.rot90(images, k=int(rotation_k), dims=(-2, -1))
                        per_rotation.append(adapter.forward_global(rotated))
                    tta = torch.stack(per_rotation, dim=1)
                tta_descriptors.append(descriptors_to_numpy(tta))
            position, xy, yaw = _to_numpy_position(batch, dataset.meta.dataset_name)
            positions.append(position)
            xys.append(xy)
            yaws.append(yaw)

    bank = FeatureBank(
        sequence_name=str(dataset.meta.sequence_name),
        sequence_dir=key,
        descriptors=np.concatenate(descriptors, axis=0).astype(np.float32, copy=False),
        position=np.concatenate(positions, axis=0).astype(np.float32, copy=False),
        xy=np.concatenate(xys, axis=0).astype(np.float32, copy=False),
        yaw=np.concatenate(yaws, axis=0).astype(np.float32, copy=False),
        indices=np.arange(len(dataset), dtype=np.int64),
        dataset=dataset,
        tta_descriptors=np.concatenate(tta_descriptors, axis=0).astype(np.float32, copy=False) if tta_descriptors else None,
    )
    if cache is not None:
        cache[key] = bank
    return bank


def _subset_bank(bank: FeatureBank, indices: Iterable[int]) -> Dict[str, np.ndarray]:
    indices = np.asarray(list(indices), dtype=np.int64)
    subset = {
        "descriptors": bank.descriptors[indices],
        "position": bank.position[indices],
        "xy": bank.xy[indices],
        "yaw": bank.yaw[indices],
        "indices": indices,
    }
    if bank.tta_descriptors is not None:
        subset["tta_descriptors"] = bank.tta_descriptors[indices]
    return subset


def _search_topk_with_optional_tta(
    db_descs: np.ndarray,
    query_descs: np.ndarray,
    query_tta_descs: np.ndarray | None,
    topk: int,
    metric: str,
    use_gpu: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if query_tta_descs is None:
        return search_topk(db_descs, query_descs, topk=int(topk), metric=metric, use_gpu=bool(use_gpu))

    query_tta_descs = np.asarray(query_tta_descs, dtype=np.float32)
    num_queries, num_rot, dim = query_tta_descs.shape
    flat = query_tta_descs.reshape(num_queries * num_rot, dim)
    search_k = min(max(int(topk) * 2, int(topk)), int(db_descs.shape[0]))
    flat_scores, flat_indices = search_topk(db_descs, flat, topk=search_k, metric=metric, use_gpu=bool(use_gpu))
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


def _load_pair_images(
    query_dataset: ProcessedSequenceDataset,
    query_indices: Sequence[int],
    candidate_dataset: ProcessedSequenceDataset,
    candidate_indices: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    query_images = torch.stack([query_dataset[int(idx)]["image"] for idx in query_indices], dim=0)
    candidate_images = torch.stack([candidate_dataset[int(idx)]["image"] for idx in candidate_indices], dim=0)
    return query_images, candidate_images


def _pose2d_from_xyyaw(xy: np.ndarray, yaw: float) -> np.ndarray:
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return np.asarray(
        [
            [c, -s, float(xy[0])],
            [s, c, float(xy[1])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _pred_pose2d_from_relative(pred_translation: np.ndarray, pred_yaw: float) -> np.ndarray:
    c = float(np.cos(pred_yaw))
    s = float(np.sin(pred_yaw))
    return np.asarray(
        [
            [c, -s, float(pred_translation[0])],
            [s, c, float(pred_translation[1])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _relative_pose_error(
    result: PairResult,
    query_xy: np.ndarray,
    query_yaw: float,
    candidate_xy: np.ndarray,
    candidate_yaw: float,
) -> tuple[float, float]:
    query_pose = _pose2d_from_xyyaw(query_xy, query_yaw)
    candidate_pose = _pose2d_from_xyyaw(candidate_xy, candidate_yaw)
    relative_gt = np.linalg.inv(candidate_pose).dot(query_pose)
    pred_pose = _pred_pose2d_from_relative(result.translation_xy, result.yaw_rad)
    err = np.linalg.inv(pred_pose).dot(relative_gt)
    err_theta = float(abs(np.arctan2(err[0, 1], err[0, 0]) / np.pi * 180.0))
    err_trans = float(np.sqrt(err[0, 2] ** 2 + err[1, 2] ** 2))
    return err_trans, err_theta


def _descriptor_pr_score(raw_score: float, metric: str) -> float:
    # BEVPlace++ thresholds nearest descriptor distance. Our PR helper assumes
    # larger scores are selected first, so L2 distance is negated.
    return -float(raw_score) if str(metric).lower() == "l2" else float(raw_score)


def _select_reranked_candidate(candidates: Sequence[Dict[str, object]], strong_min_inliers: int) -> Dict[str, object] | None:
    if not candidates:
        return None
    strong = [
        item
        for item in candidates
        if bool(getattr(item["result"], "pose_valid", False))
        and int(getattr(item["result"], "num_inliers", 0)) >= int(strong_min_inliers)
    ]
    if strong:
        return max(
            strong,
            key=lambda item: (
                int(getattr(item["result"], "num_inliers", 0)),
                float(getattr(item["result"], "score", 0.0)),
                -int(item["rank"]),
            ),
        )
    return min(candidates, key=lambda item: int(item["rank"]))


def _min_distances_to_database(query_position: np.ndarray, db_position: np.ndarray, chunk_size: int = 1024) -> np.ndarray:
    query_position = np.asarray(query_position, dtype=np.float32)
    db_position = np.asarray(db_position, dtype=np.float32)
    out = np.empty((query_position.shape[0],), dtype=np.float32)
    for start in range(0, query_position.shape[0], int(chunk_size)):
        end = min(start + int(chunk_size), query_position.shape[0])
        diff = query_position[start:end, None, :] - db_position[None, :, :]
        out[start:end] = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))
    return out


def _with_mean_row(rows: Sequence[Dict[str, object]], metric_keys: Sequence[str]) -> List[Dict[str, object]]:
    rows = [dict(row) for row in rows]
    if not rows:
        return rows
    mean_row: Dict[str, object] = {"sequence": "Mean", "database": "-"}
    for key in metric_keys:
        mean_row[key] = mean_or_zero([float(row[key]) for row in rows if key in row and row[key] is not None])
    return rows + [mean_row]


def _filter_specs_by_sequence_names(
    specs: Sequence[RetrievalSpec],
    sequence_names: Sequence[str] | None,
) -> List[RetrievalSpec]:
    if not sequence_names:
        return list(specs)
    wanted = {str(name).strip().lower() for name in sequence_names if str(name).strip()}
    if not wanted:
        return list(specs)
    filtered = [spec for spec in specs if str(spec.sequence_name).strip().lower() in wanted]
    found = {str(spec.sequence_name).strip().lower() for spec in filtered}
    missing = sorted(wanted - found)
    if missing:
        available = [str(spec.sequence_name) for spec in specs]
        raise ValueError(
            f"Requested sequence_names not found: {missing}. Available: {available}"
        )
    return filtered


def _build_specs(dataset_name: str, processed_root: str | Path, positive_radius_m: float) -> List[RetrievalSpec]:
    dataset_name = str(dataset_name).lower()
    if dataset_name == "kitti":
        specs = []
        for sequence in ["00", "02", "05", "06", "08"]:
            spec = build_kitti_place_spec(processed_root, sequence=sequence, positive_radius_m=float(positive_radius_m))
            specs.append(
                RetrievalSpec(
                    dataset_name="kitti",
                    sequence_name=spec.sequence_name,
                    database_sequence_name=spec.sequence_name,
                    query_sequence_dir=spec.sequence_dir,
                    database_sequence_dir=spec.sequence_dir,
                    query_indices=spec.query_indices,
                    database_indices=spec.db_indices,
                    positive_radius_m=spec.positive_radius_m,
                )
            )
        return specs
    if dataset_name == "nclt":
        return [
            RetrievalSpec(
                dataset_name="nclt",
                sequence_name=spec.query_sequence_name,
                database_sequence_name=spec.database_sequence_name,
                query_sequence_dir=spec.query_sequence_dir,
                database_sequence_dir=spec.database_sequence_dir,
                query_indices=spec.query_indices,
                database_indices=spec.database_indices,
                positive_radius_m=spec.positive_radius_m,
            )
            for spec in build_nclt_place_specs(processed_root, positive_radius_m=float(positive_radius_m))
        ]
    if dataset_name == "pohang":
        sequence_pairs = [
            ("pohang01", "pohang00"),
            ("pohang00", "pohang01"),
            ("pohang00", "pohang02"),
            ("pohang00", "pohang03"),
            ("pohang00", "pohang04"),
            ("pohang00", "pohang05"),
        ]
        return [
            RetrievalSpec(
                dataset_name="pohang",
                sequence_name=spec.query_sequence_name,
                database_sequence_name=spec.database_sequence_name,
                query_sequence_dir=spec.query_sequence_dir,
                database_sequence_dir=spec.database_sequence_dir,
                query_indices=spec.query_indices,
                database_indices=spec.database_indices,
                positive_radius_m=spec.positive_radius_m,
            )
            for spec in build_pohang_place_specs(
                processed_root,
                sequence_pairs=sequence_pairs,
                positive_radius_m=float(positive_radius_m),
            )
        ]
    if dataset_name == "usvinland":
        return [
            RetrievalSpec(
                dataset_name="usvinland",
                sequence_name=spec.query_sequence_name,
                database_sequence_name=spec.database_sequence_name,
                query_sequence_dir=spec.query_sequence_dir,
                database_sequence_dir=spec.database_sequence_dir,
                query_indices=spec.query_indices,
                database_indices=spec.database_indices,
                positive_radius_m=spec.positive_radius_m,
            )
            for spec in build_usvinland_place_specs(processed_root, positive_radius_m=float(positive_radius_m))
        ]
    raise ValueError(f"Unsupported backend dataset: {dataset_name}")


def _pair_eval(
    adapter,
    backend: SparseRansacBackend,
    query_bank: FeatureBank,
    db_bank: FeatureBank,
    pair_query_indices: Sequence[int],
    pair_db_indices: Sequence[int],
    pair_batch_size: int,
    device: torch.device,
) -> List[PairResult]:
    results: List[PairResult] = []
    meters_per_pixel = float(getattr(query_bank.dataset.meta, "meters_per_pixel", 0.4))
    for start in range(0, len(pair_query_indices), int(pair_batch_size)):
        end = min(start + int(pair_batch_size), len(pair_query_indices))
        query_images, candidate_images = _load_pair_images(
            query_bank.dataset,
            pair_query_indices[start:end],
            db_bank.dataset,
            pair_db_indices[start:end],
        )
        with torch.no_grad():
            _, _, query_local, candidate_local = adapter.forward_pair_features(
                query_images.to(device, non_blocking=True),
                candidate_images.to(device, non_blocking=True),
            )
        batch_results = backend.solve_batch(
            query_images=query_images.to(device, non_blocking=True),
            candidate_images=candidate_images.to(device, non_blocking=True),
            query_local=query_local,
            candidate_local=candidate_local,
            meters_per_pixel=meters_per_pixel,
        )
        results.extend(batch_results)
        if end == len(pair_query_indices) or end == int(pair_batch_size) or end % max(int(pair_batch_size) * 100, 100) == 0:
            print(f"[Backend] pair eval {end}/{len(pair_query_indices)}", flush=True)
    return results


def _rerank_eval(
    adapter,
    backend: SparseRansacBackend,
    query_bank: FeatureBank,
    db_bank: FeatureBank,
    candidate_meta: Sequence[Dict[str, object]],
    pair_batch_size: int,
    device: torch.device,
) -> List[PairResult]:
    if not candidate_meta:
        return []
    max_rank = max(int(item.get("rank", 0)) for item in candidate_meta)
    if max_rank <= 0 or not hasattr(adapter, "forward_local_features"):
        return _pair_eval(
            adapter,
            backend,
            query_bank,
            db_bank,
            [int(item["query_index"]) for item in candidate_meta],
            [int(item["db_index"]) for item in candidate_meta],
            pair_batch_size=pair_batch_size,
            device=device,
        )

    results_by_meta = [PairResult.empty() for _ in candidate_meta]
    meters_per_pixel = float(getattr(query_bank.dataset.meta, "meters_per_pixel", 0.4))
    groups: Dict[int, List[int]] = {}
    for meta_i, item in enumerate(candidate_meta):
        groups.setdefault(int(item["query_index"]), []).append(meta_i)

    processed = 0
    for q_idx, meta_indices in groups.items():
        query_image_cpu = query_bank.dataset[int(q_idx)]["image"]
        query_image = query_image_cpu.unsqueeze(0).to(device, non_blocking=True)
        with torch.no_grad():
            query_local_one = adapter.forward_local_features(query_image)[0].detach().cpu()

        for start in range(0, len(meta_indices), int(pair_batch_size)):
            chunk = meta_indices[start : start + int(pair_batch_size)]
            candidate_images_cpu = torch.stack(
                [db_bank.dataset[int(candidate_meta[meta_i]["db_index"])]["image"] for meta_i in chunk],
                dim=0,
            )
            with torch.no_grad():
                candidate_local = adapter.forward_local_features(candidate_images_cpu.to(device, non_blocking=True)).detach().cpu()
            query_images = query_image_cpu.unsqueeze(0).repeat(len(chunk), 1, 1, 1)
            query_local = query_local_one.unsqueeze(0).repeat(len(chunk), 1, 1, 1)
            batch_results = backend.solve_batch(
                query_images=query_images,
                candidate_images=candidate_images_cpu,
                query_local=query_local,
                candidate_local=candidate_local,
                meters_per_pixel=meters_per_pixel,
            )
            for meta_i, result in zip(chunk, batch_results):
                results_by_meta[int(meta_i)] = result
            processed += len(chunk)
            if processed == len(candidate_meta) or processed <= max(int(pair_batch_size), 1) or processed % max(int(pair_batch_size) * 100, 100) == 0:
                print(f"[Backend] rerank eval {processed}/{len(candidate_meta)}", flush=True)
    return results_by_meta


def evaluate_loop(
    adapter,
    backend: SparseRansacBackend,
    dataset_name: str,
    processed_root: str | Path,
    output_dir: str | Path,
    device: torch.device,
    image_size: int = 201,
    eval_batch_size: int = 64,
    pair_batch_size: int = 2,
    num_workers: int = 0,
    retrieval_metric: str = "l2",
    positive_radius_m: float = 5.0,
    negative_radius_m: float = 15.0,
    kitti_loader_mode: str = "bevplace2_eval_gray3",
    faiss_gpu: bool = True,
    cache: Dict[str, FeatureBank] | None = None,
    sequence_names: Sequence[str] | None = None,
    max_sequences: int | None = None,
    max_pairs_per_sequence: int | None = None,
    rerank_top_k: int = 1,
    rerank_top_v: int = 1,
    rerank_strong_min_inliers: int = 8,
) -> Dict:
    output_dir = ensure_dir(output_dir)
    specs = _build_specs(dataset_name, processed_root, positive_radius_m=positive_radius_m)
    specs = _filter_specs_by_sequence_names(specs, sequence_names)
    if max_sequences is not None and int(max_sequences) > 0:
        specs = specs[: int(max_sequences)]
    if cache is None:
        cache = {}
    per_sequence: List[Dict[str, object]] = []
    all_pairs: List[Dict[str, object]] = []

    for spec in specs:
        print(f"[Loop] {dataset_name} seq={spec.sequence_name} db={spec.database_sequence_name} descriptors...", flush=True)
        db_bank = _extract_feature_bank(
            adapter,
            spec.database_sequence_dir,
            image_size=image_size,
            batch_size=eval_batch_size,
            device=device,
            cache=cache,
            kitti_loader_mode=kitti_loader_mode,
            num_workers=num_workers,
            include_tta=False,
        )
        query_bank = _extract_feature_bank(
            adapter,
            spec.query_sequence_dir,
            image_size=image_size,
            batch_size=eval_batch_size,
            device=device,
            cache=cache,
            kitti_loader_mode=kitti_loader_mode,
            num_workers=num_workers,
            include_tta=bool(getattr(adapter, "query_uses_tta", False)),
        )
        db = _subset_bank(db_bank, spec.database_indices)
        query = _subset_bank(query_bank, spec.query_indices)
        retrieve_k = max(1, int(rerank_top_k))
        verify_v = max(1, min(int(rerank_top_v), retrieve_k))
        retrieval_scores, predictions = _search_topk_with_optional_tta(
            db["descriptors"],
            query["descriptors"],
            query.get("tta_descriptors", None),
            topk=retrieve_k,
            metric=retrieval_metric,
            use_gpu=bool(faiss_gpu),
        )

        min_gt_distances = _min_distances_to_database(query["position"], db["position"])
        has_positive = min_gt_distances < float(positive_radius_m)
        total_positive_queries = int(np.sum(has_positive))
        pair_query_indices: List[int] = []
        pair_db_indices: List[int] = []
        labels: List[int] = []
        scores: List[float] = []
        details: List[Dict[str, object]] = []
        candidate_meta: List[Dict[str, object]] = []
        candidates_by_row: Dict[int, List[Dict[str, object]]] = {}
        query_lookup = {int(idx): pos for pos, idx in enumerate(query["indices"])}
        db_lookup = {int(idx): pos for pos, idx in enumerate(db["indices"])}
        for qi in range(int(predictions.shape[0])):
            top1_local = int(predictions[qi, 0])
            top1_dist = float(np.linalg.norm(db["position"][top1_local] - query["position"][qi]))
            label = int(top1_dist < float(positive_radius_m))
            pr_score = _descriptor_pr_score(float(retrieval_scores[qi, 0]), retrieval_metric)
            q_idx = int(query["indices"][qi])
            top1_d_idx = int(db["indices"][top1_local])
            labels.append(label)
            scores.append(float(pr_score))
            row_idx = len(details)
            details.append(
                {
                    "dataset": str(dataset_name),
                    "sequence": spec.sequence_name,
                    "database": spec.database_sequence_name,
                    "query_index": q_idx,
                    "db_index": top1_d_idx,
                    "top1_db_index": top1_d_idx,
                    "label": int(label),
                    "has_positive": int(has_positive[qi]),
                    "nearest_gt_db_distance_m": float(min_gt_distances[qi]),
                    "retrieval_distance_m": top1_dist,
                    "descriptor_raw_score": float(retrieval_scores[qi, 0]),
                    "score": float(pr_score),
                }
            )
            candidates_by_row[row_idx] = []
            seen_candidates: set[int] = set()
            for rank in range(verify_v):
                cand_local = int(predictions[qi, rank])
                if cand_local in seen_candidates:
                    continue
                seen_candidates.add(cand_local)
                d_idx = int(db["indices"][cand_local])
                dist = float(np.linalg.norm(db["position"][cand_local] - query["position"][qi]))
                cand = {
                    "row_idx": row_idx,
                    "rank": int(rank),
                    "query_index": q_idx,
                    "db_index": d_idx,
                    "db_local": cand_local,
                    "descriptor_raw_score": float(retrieval_scores[qi, rank]),
                    "retrieval_distance_m": dist,
                    "label": int(dist < float(positive_radius_m)),
                }
                candidates_by_row[row_idx].append(cand)
                candidate_meta.append(cand)
                pair_query_indices.append(q_idx)
                pair_db_indices.append(d_idx)

        print(
            f"[Loop] {dataset_name} seq={spec.sequence_name} pairs={len(details)} verify_pairs={len(pair_query_indices)} "
            f"top1_tp={int(sum(labels))} gt_positive_queries={total_positive_queries}",
            flush=True,
        )
        if max_pairs_per_sequence is not None and int(max_pairs_per_sequence) > 0:
            limit = int(max_pairs_per_sequence)
            pair_query_indices = pair_query_indices[:limit]
            pair_db_indices = pair_db_indices[:limit]
            candidate_meta = candidate_meta[:limit]
            kept_rows = {int(item["row_idx"]) for item in candidate_meta}
            details = [row for row_i, row in enumerate(details) if row_i in kept_rows]
            labels = [label for row_i, label in enumerate(labels) if row_i in kept_rows]
            scores = [score for row_i, score in enumerate(scores) if row_i in kept_rows]
            candidates_by_row = {
                new_i: [cand for cand in candidates_by_row[old_i] if cand in candidate_meta]
                for new_i, old_i in enumerate(sorted(kept_rows))
            }
            for new_i, old_i in enumerate(sorted(kept_rows)):
                for cand in candidates_by_row[new_i]:
                    cand["row_idx"] = new_i
            total_positive_queries = int(sum(int(row["has_positive"]) for row in details))
        results = _rerank_eval(
            adapter,
            backend,
            query_bank,
            db_bank,
            candidate_meta,
            pair_batch_size=pair_batch_size,
            device=device,
        )

        for cand, result in zip(candidate_meta, results):
            cand["result"] = result

        for row_idx, row in enumerate(details):
            selected = _select_reranked_candidate(candidates_by_row.get(row_idx, []), rerank_strong_min_inliers)
            if selected is None:
                row["selected_rank"] = 0
                row["selected_by_rerank"] = 0
                row["selected_db_index"] = int(row["top1_db_index"])
                row["selected_label"] = int(row["label"])
                row["verified_candidates"] = 0
                row["backend_score"] = 0.0
                row["pose_valid"] = 0
                row["num_inliers"] = 0
                row["num_matches"] = 0
                row["inlier_mean_residual_m"] = None
                row["e_t_m"] = None
                row["e_r_deg"] = None
                continue
            result = selected.get("result", PairResult.empty())
            q_local = query_lookup[int(row["query_index"])]
            d_local = db_lookup[int(selected["db_index"])]
            row["db_index"] = int(selected["db_index"])
            row["selected_db_index"] = int(selected["db_index"])
            row["selected_rank"] = int(selected["rank"])
            row["selected_by_rerank"] = int(int(selected["rank"]) != 0)
            row["selected_label"] = int(selected["label"])
            row["selected_descriptor_raw_score"] = float(selected["descriptor_raw_score"])
            row["selected_retrieval_distance_m"] = float(selected["retrieval_distance_m"])
            row["verified_candidates"] = int(len(candidates_by_row.get(row_idx, [])))
            row["backend_score"] = float(result.score)
            row["pose_valid"] = int(result.pose_valid)
            row["num_inliers"] = int(result.num_inliers)
            row["num_matches"] = int(result.num_matches)
            row["inlier_mean_residual_m"] = (
                float(result.inlier_mean_residual_m) if np.isfinite(result.inlier_mean_residual_m) else None
            )
            if result.pose_valid:
                err_t, err_r = _relative_pose_error(
                    result,
                    query_xy=query["xy"][q_local],
                    query_yaw=float(query["yaw"][q_local]),
                    candidate_xy=db["xy"][d_local],
                    candidate_yaw=float(db["yaw"][d_local]),
                )
                row["e_t_m"] = float(err_t)
                row["e_r_deg"] = float(err_r)
            else:
                row["e_t_m"] = None
                row["e_r_deg"] = None

        labels_np = np.asarray(labels, dtype=np.int64)
        scores_np = np.asarray(scores, dtype=np.float32)
        curve = precision_recall_curve(scores_np, labels_np, total_positives=total_positive_queries)
        max_f1, threshold = best_f1_and_threshold(curve["precision"], curve["recall"], curve["thresholds"])
        true_positive_pose_rows = [
            row
            for row in details
            if int(row.get("selected_label", row["label"])) == 1
            and int(row["pose_valid"]) == 1
            and row["e_t_m"] is not None
            and row["e_r_deg"] is not None
        ]
        summary = {
            "dataset": str(dataset_name),
            "sequence": spec.sequence_name,
            "database": spec.database_sequence_name,
            "AP": area_under_pr(curve["precision"], curve["recall"]),
            "MaxF1": max_f1,
            "MaxRecall@100P": max_recall_at_precision(curve["precision"], curve["recall"], target_precision=1.0),
            "e_t_m": mean_or_zero([float(row["e_t_m"]) for row in true_positive_pose_rows]),
            "e_r_deg": mean_or_zero([float(row["e_r_deg"]) for row in true_positive_pose_rows]),
            "max_f1_threshold": float(threshold),
            "max_f1_descriptor_distance": float(-threshold) if str(retrieval_metric).lower() == "l2" else float(threshold),
            "num_pairs": int(len(details)),
            "num_positive_pairs": int(sum(labels)),
            "num_selected_positive_pairs": int(sum(int(row.get("selected_label", row["label"])) for row in details)),
            "num_positive_queries": int(total_positive_queries),
            "num_true_positive_pose_queries": int(len(true_positive_pose_rows)),
            "rerank_top_k": int(retrieve_k),
            "rerank_top_v": int(verify_v),
            "rerank_strong_min_inliers": int(rerank_strong_min_inliers),
        }
        per_sequence.append(summary)
        all_pairs.extend(details)
        print(
            f"[Loop] {dataset_name} seq={spec.sequence_name} AP={summary['AP']:.4f} "
            f"F1={summary['MaxF1']:.4f} R100P={summary['MaxRecall@100P']:.4f} "
            f"e_t={summary['e_t_m']:.3f} e_r={summary['e_r_deg']:.3f}",
            flush=True,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "dataset": str(dataset_name),
        "per_sequence": per_sequence,
        "Mean AP": mean_or_zero([float(row["AP"]) for row in per_sequence]),
        "Mean MaxF1": mean_or_zero([float(row["MaxF1"]) for row in per_sequence]),
        "Mean MaxRecall@100P": mean_or_zero([float(row["MaxRecall@100P"]) for row in per_sequence]),
        "Mean e_t_m": mean_or_zero([float(row["e_t_m"]) for row in per_sequence]),
        "Mean e_r_deg": mean_or_zero([float(row["e_r_deg"]) for row in per_sequence]),
            "notes": "BEVPlace2-main aligned protocol: KITTI positives use raw 3D pose distance [x,y,z]; KITTI pose error uses BEV ground-plane [x,z]+yaw. Loop PR/AP/MaxF1/MaxRecall@100P threshold the nearest top-1 descriptor distance; e_t/e_r use the selected top-V geometric reranking result.",
    }
    save_json(output_dir / "paper_loop_v4.json", payload)
    save_tsv(output_dir / "paper_loop_v4.tsv", _with_mean_row(per_sequence, ["AP", "MaxF1", "MaxRecall@100P", "e_t_m", "e_r_deg"]))
    save_tsv(output_dir / "paper_loop_pairs_v4.tsv", all_pairs)
    return payload


def evaluate_global_loc(
    adapter,
    backend: SparseRansacBackend,
    dataset_name: str,
    processed_root: str | Path,
    output_dir: str | Path,
    device: torch.device,
    image_size: int = 201,
    eval_batch_size: int = 64,
    pair_batch_size: int = 2,
    num_workers: int = 0,
    retrieval_metric: str = "l2",
    positive_radius_m: float = 5.0,
    success_translation_m: float = 2.0,
    success_rotation_deg: float = 5.0,
    kitti_loader_mode: str = "bevplace2_eval_gray3",
    faiss_gpu: bool = True,
    cache: Dict[str, FeatureBank] | None = None,
    sequence_names: Sequence[str] | None = None,
    max_sequences: int | None = None,
    max_pairs_per_sequence: int | None = None,
    rerank_top_k: int = 1,
    rerank_top_v: int = 1,
    rerank_strong_min_inliers: int = 8,
) -> Dict:
    output_dir = ensure_dir(output_dir)
    specs = _build_specs(dataset_name, processed_root, positive_radius_m=positive_radius_m)
    specs = _filter_specs_by_sequence_names(specs, sequence_names)
    if max_sequences is not None and int(max_sequences) > 0:
        specs = specs[: int(max_sequences)]
    if cache is None:
        cache = {}
    per_sequence: List[Dict[str, object]] = []
    all_pairs: List[Dict[str, object]] = []

    for spec in specs:
        print(f"[GlobalLoc] {dataset_name} seq={spec.sequence_name} db={spec.database_sequence_name} descriptors...", flush=True)
        db_bank = _extract_feature_bank(
            adapter,
            spec.database_sequence_dir,
            image_size=image_size,
            batch_size=eval_batch_size,
            device=device,
            cache=cache,
            kitti_loader_mode=kitti_loader_mode,
            num_workers=num_workers,
            include_tta=False,
        )
        query_bank = _extract_feature_bank(
            adapter,
            spec.query_sequence_dir,
            image_size=image_size,
            batch_size=eval_batch_size,
            device=device,
            cache=cache,
            kitti_loader_mode=kitti_loader_mode,
            num_workers=num_workers,
            include_tta=bool(getattr(adapter, "query_uses_tta", False)),
        )
        db = _subset_bank(db_bank, spec.database_indices)
        query = _subset_bank(query_bank, spec.query_indices)
        retrieve_k = max(1, int(rerank_top_k))
        verify_v = max(1, min(int(rerank_top_v), retrieve_k))
        retrieval_scores, predictions = _search_topk_with_optional_tta(
            db["descriptors"],
            query["descriptors"],
            query.get("tta_descriptors", None),
            topk=retrieve_k,
            metric=retrieval_metric,
            use_gpu=bool(faiss_gpu),
        )

        pair_query_indices: List[int] = []
        pair_db_indices: List[int] = []
        eligible_rows: List[int] = []
        details: List[Dict[str, object]] = []
        candidate_meta: List[Dict[str, object]] = []
        candidates_by_row: Dict[int, List[Dict[str, object]]] = {}
        query_lookup = {int(idx): pos for pos, idx in enumerate(query["indices"])}
        db_lookup = {int(idx): pos for pos, idx in enumerate(db["indices"])}

        for qi in range(int(predictions.shape[0])):
            top1_local = int(predictions[qi, 0])
            distances = np.linalg.norm(db["position"] - query["position"][qi : qi + 1], axis=1)
            positives = np.where(distances < float(positive_radius_m))[0]
            has_positive = positives.size > 0
            positives_set = set(int(v) for v in positives.tolist())
            retrieval_hit = int(has_positive and top1_local in positives_set)
            q_idx = int(query["indices"][qi])
            top1_d_idx = int(db["indices"][top1_local])
            row_idx = len(details)
            row = {
                "dataset": str(dataset_name),
                "sequence": spec.sequence_name,
                "database": spec.database_sequence_name,
                "query_index": q_idx,
                "db_index": top1_d_idx,
                "top1_db_index": top1_d_idx,
                "descriptor_score": float(retrieval_scores[qi, 0]),
                "nearest_gt_db_distance_m": float(np.min(distances)) if distances.size else float("inf"),
                "retrieval_distance_m": float(distances[top1_local]),
                "has_positive": int(has_positive),
                "retrieval_hit": retrieval_hit,
                "pose_valid": 0,
                "e_t_m": None,
                "e_r_deg": None,
                "success": 0,
            }
            details.append(row)
            candidates_by_row[row_idx] = []
            if has_positive:
                eligible_rows.append(len(details) - 1)
                seen_candidates: set[int] = set()
                for rank in range(verify_v):
                    cand_local = int(predictions[qi, rank])
                    if cand_local in seen_candidates:
                        continue
                    seen_candidates.add(cand_local)
                    d_idx = int(db["indices"][cand_local])
                    cand = {
                        "row_idx": row_idx,
                        "rank": int(rank),
                        "query_index": q_idx,
                        "db_index": d_idx,
                        "db_local": cand_local,
                        "descriptor_raw_score": float(retrieval_scores[qi, rank]),
                        "retrieval_distance_m": float(distances[cand_local]),
                        "label": int(cand_local in positives_set),
                    }
                    candidates_by_row[row_idx].append(cand)
                    candidate_meta.append(cand)
                    pair_query_indices.append(q_idx)
                    pair_db_indices.append(d_idx)

        print(
            f"[GlobalLoc] {dataset_name} seq={spec.sequence_name} queries={len(details)} "
            f"eligible={len(eligible_rows)} verify_pairs={len(pair_query_indices)}",
            flush=True,
        )
        if max_pairs_per_sequence is not None and int(max_pairs_per_sequence) > 0:
            limit = int(max_pairs_per_sequence)
            pair_query_indices = pair_query_indices[:limit]
            pair_db_indices = pair_db_indices[:limit]
            candidate_meta = candidate_meta[:limit]
            kept_rows = sorted({int(item["row_idx"]) for item in candidate_meta})
            eligible_rows = [row_i for row_i in eligible_rows if row_i in set(kept_rows)]
            candidates_by_row = {
                row_i: [cand for cand in candidates_by_row[row_i] if cand in candidate_meta]
                for row_i in kept_rows
            }
        results = _rerank_eval(
            adapter,
            backend,
            query_bank,
            db_bank,
            candidate_meta,
            pair_batch_size=pair_batch_size,
            device=device,
        )

        for cand, result in zip(candidate_meta, results):
            cand["result"] = result

        for row_idx in eligible_rows:
            row = details[int(row_idx)]
            selected = _select_reranked_candidate(candidates_by_row.get(int(row_idx), []), rerank_strong_min_inliers)
            if selected is None:
                row["selected_rank"] = 0
                row["selected_by_rerank"] = 0
                row["selected_db_index"] = int(row["top1_db_index"])
                row["selected_label"] = 0
                row["verified_candidates"] = 0
                continue
            result = selected.get("result", PairResult.empty())
            q_local = query_lookup[int(row["query_index"])]
            d_local = db_lookup[int(selected["db_index"])]
            row["db_index"] = int(selected["db_index"])
            row["selected_db_index"] = int(selected["db_index"])
            row["selected_rank"] = int(selected["rank"])
            row["selected_by_rerank"] = int(int(selected["rank"]) != 0)
            row["selected_label"] = int(selected["label"])
            row["selected_descriptor_raw_score"] = float(selected["descriptor_raw_score"])
            row["selected_retrieval_distance_m"] = float(selected["retrieval_distance_m"])
            row["verified_candidates"] = int(len(candidates_by_row.get(int(row_idx), [])))
            row["backend_score"] = float(result.score)
            row["pose_valid"] = int(result.pose_valid)
            row["num_inliers"] = int(result.num_inliers)
            row["num_matches"] = int(result.num_matches)
            if result.pose_valid:
                err_t, err_r = _relative_pose_error(
                    result,
                    query_xy=query["xy"][q_local],
                    query_yaw=float(query["yaw"][q_local]),
                    candidate_xy=db["xy"][d_local],
                    candidate_yaw=float(db["yaw"][d_local]),
                )
                row["e_t_m"] = float(err_t)
                row["e_r_deg"] = float(err_r)
                row["success"] = int(err_t < float(success_translation_m) and err_r < float(success_rotation_deg))

        eligible = [row for row in details if int(row["has_positive"]) == 1]
        success_rows = [row for row in eligible if int(row["success"]) == 1]
        summary = {
            "dataset": str(dataset_name),
            "sequence": spec.sequence_name,
            "database": spec.database_sequence_name,
            "Recall@1": float(sum(int(row["retrieval_hit"]) for row in eligible)) / float(max(len(eligible), 1)),
            "RerankedRecall@1": float(sum(int(row.get("selected_label", 0)) for row in eligible)) / float(max(len(eligible), 1)),
            "SuccessRate": float(len(success_rows)) / float(max(len(eligible), 1)),
            "MeanTransErr": mean_or_zero([float(row["e_t_m"]) for row in success_rows if row["e_t_m"] is not None]),
            "MeanRotErr": mean_or_zero([float(row["e_r_deg"]) for row in success_rows if row["e_r_deg"] is not None]),
            "AllPositiveQueries": int(len(eligible)),
            "SuccessfulQueries": int(len(success_rows)),
            "rerank_top_k": int(retrieve_k),
            "rerank_top_v": int(verify_v),
            "rerank_strong_min_inliers": int(rerank_strong_min_inliers),
        }
        per_sequence.append(summary)
        all_pairs.extend(details)
        print(
            f"[GlobalLoc] {dataset_name} seq={spec.sequence_name} R@1={summary['Recall@1']:.4f} "
            f"SR={summary['SuccessRate']:.4f} e_t={summary['MeanTransErr']:.3f} e_r={summary['MeanRotErr']:.3f}",
            flush=True,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "dataset": str(dataset_name),
        "per_sequence": per_sequence,
        "Mean Recall@1": mean_or_zero([float(row["Recall@1"]) for row in per_sequence]),
        "Mean SuccessRate": mean_or_zero([float(row["SuccessRate"]) for row in per_sequence]),
        "MeanTransErr": mean_or_zero([float(row["MeanTransErr"]) for row in per_sequence]),
        "MeanRotErr": mean_or_zero([float(row["MeanRotErr"]) for row in per_sequence]),
        "success_translation_m": float(success_translation_m),
        "success_rotation_deg": float(success_rotation_deg),
            "notes": "BEVPlace2-main aligned protocol: KITTI positives use raw 3D pose distance [x,y,z]; KITTI pose error uses BEV ground-plane [x,z]+yaw. Recall@1 reports top-1 global retrieval; SuccessRate uses top-V geometric reranking with fallback to top-1 when no strong geometric candidate exists.",
    }
    save_json(output_dir / "paper_global_loc_v4.json", payload)
    save_tsv(
        output_dir / "paper_global_loc_v4.tsv",
        _with_mean_row(per_sequence, ["Recall@1", "RerankedRecall@1", "SuccessRate", "MeanTransErr", "MeanRotErr"]),
    )
    save_tsv(output_dir / "paper_global_loc_pairs_v4.tsv", all_pairs)
    return payload


def benchmark_runtime(
    adapter,
    backend: SparseRansacBackend,
    dataset_name: str,
    processed_root: str | Path,
    output_dir: str | Path,
    device: torch.device,
    image_size: int = 201,
    eval_batch_size: int = 64,
    num_workers: int = 0,
    retrieval_metric: str = "l2",
    positive_radius_m: float = 5.0,
    kitti_loader_mode: str = "bevplace2_eval_gray3",
    faiss_gpu: bool = True,
    warmup: int = 10,
    timed_queries: int = 50,
    cache: Dict[str, FeatureBank] | None = None,
    sequence_names: Sequence[str] | None = None,
    force_query_tta: bool | None = None,
) -> Dict:
    output_dir = ensure_dir(output_dir)
    specs = _build_specs(dataset_name, processed_root, positive_radius_m=positive_radius_m)
    specs = _filter_specs_by_sequence_names(specs, sequence_names)
    spec = specs[-1 if str(dataset_name).lower() == "kitti" and len(specs) > 1 else 0]
    if cache is None:
        cache = {}
    use_query_tta = bool(getattr(adapter, "query_uses_tta", False)) if force_query_tta is None else bool(force_query_tta)
    db_bank = _extract_feature_bank(
        adapter,
        spec.database_sequence_dir,
        image_size=image_size,
        batch_size=eval_batch_size,
        device=device,
        cache=cache,
        kitti_loader_mode=kitti_loader_mode,
        num_workers=num_workers,
        include_tta=False,
    )
    query_bank = _extract_feature_bank(
        adapter,
        spec.query_sequence_dir,
        image_size=image_size,
        batch_size=eval_batch_size,
        device=device,
        cache=cache,
        kitti_loader_mode=kitti_loader_mode,
        num_workers=num_workers,
        include_tta=use_query_tta,
    )
    db = _subset_bank(db_bank, spec.database_indices)
    index = _DescriptorIndex(db["descriptors"], metric=retrieval_metric, use_gpu=bool(faiss_gpu))
    query_indices = list(spec.query_indices[: min(int(timed_queries), len(spec.query_indices))])
    if not query_indices:
        raise RuntimeError(f"No runtime queries for {dataset_name} {spec.sequence_name}.")

    frontend_times: List[float] = []
    retrieval_times: List[float] = []
    verification_times: List[float] = []
    e2e_times: List[float] = []

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device=device)

    def merge_single_query_tta(raw_scores: np.ndarray, raw_indices: np.ndarray) -> np.ndarray:
        prefer_low = str(retrieval_metric).lower() == "l2"
        merged: dict[int, float] = {}
        for row_scores, row_indices in zip(raw_scores, raw_indices):
            for score, index_value in zip(row_scores, row_indices):
                index_value = int(index_value)
                score = float(score)
                if index_value < 0:
                    continue
                if index_value not in merged or (score < merged[index_value] if prefer_low else score > merged[index_value]):
                    merged[index_value] = score
        ordered = sorted(merged.items(), key=lambda item: item[1], reverse=not prefer_low)
        best_index = int(ordered[0][0]) if ordered else 0
        return np.asarray([[best_index]], dtype=np.int64)

    total_iters = int(warmup) + len(query_indices)
    for step in range(total_iters):
        query_index = int(query_indices[step % len(query_indices)])
        query_image = query_bank.dataset[query_index]["image"].unsqueeze(0).to(device)

        sync()
        start = time.perf_counter()
        with torch.no_grad():
            if use_query_tta:
                if hasattr(adapter, "forward_global_tta"):
                    query_desc = descriptors_to_numpy(adapter.forward_global_tta(query_image)[0])
                else:
                    per_rotation = [adapter.forward_global(query_image)]
                    for rotation_k in (1, 2, 3):
                        rotated = torch.rot90(query_image, k=int(rotation_k), dims=(-2, -1))
                        per_rotation.append(adapter.forward_global(rotated))
                    query_desc = descriptors_to_numpy(torch.cat(per_rotation, dim=0))
            else:
                query_desc = descriptors_to_numpy(adapter.forward_global(query_image))
        sync()
        frontend_ms = (time.perf_counter() - start) * 1000.0

        sync()
        start = time.perf_counter()
        if use_query_tta:
            raw_scores, raw_pred = index.search(query_desc, topk=min(2, int(db["descriptors"].shape[0])))
            pred = merge_single_query_tta(raw_scores, raw_pred)
        else:
            _, pred = index.search(query_desc, topk=1)
        sync()
        retrieval_ms = (time.perf_counter() - start) * 1000.0

        db_index = int(db["indices"][int(pred[0, 0])])
        candidate_image = db_bank.dataset[db_index]["image"].unsqueeze(0).to(device)
        sync()
        start = time.perf_counter()
        with torch.no_grad():
            _, _, query_local, candidate_local = adapter.forward_pair_features(query_image, candidate_image)
            _ = backend.solve_batch(query_image, candidate_image, query_local, candidate_local, db_bank.dataset.meta.meters_per_pixel)
        sync()
        verification_ms = (time.perf_counter() - start) * 1000.0

        sync()
        start = time.perf_counter()
        with torch.no_grad():
            if use_query_tta:
                if hasattr(adapter, "forward_global_tta"):
                    query_desc_full = descriptors_to_numpy(adapter.forward_global_tta(query_image)[0])
                else:
                    per_rotation = [adapter.forward_global(query_image)]
                    for rotation_k in (1, 2, 3):
                        rotated = torch.rot90(query_image, k=int(rotation_k), dims=(-2, -1))
                        per_rotation.append(adapter.forward_global(rotated))
                    query_desc_full = descriptors_to_numpy(torch.cat(per_rotation, dim=0))
                raw_scores_full, raw_pred_full = index.search(query_desc_full, topk=min(2, int(db["descriptors"].shape[0])))
                pred_full = merge_single_query_tta(raw_scores_full, raw_pred_full)
            else:
                query_desc_full = descriptors_to_numpy(adapter.forward_global(query_image))
                _, pred_full = index.search(query_desc_full, topk=1)
            candidate_image_full = db_bank.dataset[int(db["indices"][int(pred_full[0, 0])])]["image"].unsqueeze(0).to(device)
            _, _, query_local_full, candidate_local_full = adapter.forward_pair_features(query_image, candidate_image_full)
            _ = backend.solve_batch(
                query_image,
                candidate_image_full,
                query_local_full,
                candidate_local_full,
                db_bank.dataset.meta.meters_per_pixel,
            )
        sync()
        e2e_ms = (time.perf_counter() - start) * 1000.0

        if step >= int(warmup):
            frontend_times.append(frontend_ms)
            retrieval_times.append(retrieval_ms)
            verification_times.append(verification_ms)
            e2e_times.append(e2e_ms)

    payload = {
        "dataset": str(dataset_name),
        "runtime_sequence": spec.sequence_name,
        "database": spec.database_sequence_name,
        "FrontendFeatureTime": summarize_runtime(frontend_times),
        "RetrievalTime": summarize_runtime(retrieval_times),
        "VerificationPoseTime": summarize_runtime(verification_times),
        "EndToEndTime": summarize_runtime(e2e_times),
        "timed_queries": int(len(query_indices)),
        "warmup": int(warmup),
        "query_tta_enabled": int(use_query_tta),
        "notes": "VerificationPoseTime includes pair feature extraction and sparse RANSAC for the retrieved top-1 candidate.",
    }
    save_json(output_dir / "paper_runtime.json", payload)
    save_tsv(
        output_dir / "paper_runtime.tsv",
        [
            {"segment": name, **stats}
            for name, stats in payload.items()
            if isinstance(stats, dict) and {"mean", "std", "p50", "p90", "fps"}.issubset(stats.keys())
        ],
    )
    return payload


def benchmark_hybrid_rerank_runtime(
    adapter,
    backend: SparseRansacBackend,
    dataset_name: str,
    processed_root: str | Path,
    output_dir: str | Path,
    device: torch.device,
    image_size: int = 201,
    eval_batch_size: int = 64,
    pair_batch_size: int = 20,
    num_workers: int = 0,
    retrieval_metric: str = "l2",
    positive_radius_m: float = 5.0,
    kitti_loader_mode: str = "bevplace2_eval_gray3",
    faiss_gpu: bool = True,
    warmup: int = 10,
    timed_queries: int = 50,
    rerank_top_k: int = 10,
    rerank_top_v: int = 5,
    rerank_strong_min_inliers: int = 8,
    cache: Dict[str, FeatureBank] | None = None,
    sequence_names: Sequence[str] | None = None,
    force_query_tta: bool | None = None,
) -> Dict:
    """Benchmark the actual hybrid Top-K/Top-V pipeline.

    Unlike ``benchmark_runtime`` this measures the deployed hybrid protocol:
    USVLoc query descriptor with 4-rotation TTA, Top-K retrieval, Top-V
    BEVPlace++ local feature extraction, and geometric reranking.
    """
    output_dir = ensure_dir(output_dir)
    specs = _build_specs(dataset_name, processed_root, positive_radius_m=positive_radius_m)
    specs = _filter_specs_by_sequence_names(specs, sequence_names)
    spec = specs[-1 if str(dataset_name).lower() == "kitti" and len(specs) > 1 else 0]
    if cache is None:
        cache = {}
    use_query_tta = bool(getattr(adapter, "query_uses_tta", False)) if force_query_tta is None else bool(force_query_tta)
    db_bank = _extract_feature_bank(
        adapter,
        spec.database_sequence_dir,
        image_size=image_size,
        batch_size=eval_batch_size,
        device=device,
        cache=cache,
        kitti_loader_mode=kitti_loader_mode,
        num_workers=num_workers,
        include_tta=False,
    )
    query_bank = _extract_feature_bank(
        adapter,
        spec.query_sequence_dir,
        image_size=image_size,
        batch_size=eval_batch_size,
        device=device,
        cache=cache,
        kitti_loader_mode=kitti_loader_mode,
        num_workers=num_workers,
        include_tta=False,
    )
    db = _subset_bank(db_bank, spec.database_indices)
    retrieve_k = max(1, int(rerank_top_k))
    verify_v = max(1, min(int(rerank_top_v), retrieve_k))
    search_k = min(max(retrieve_k * 2, retrieve_k), int(db["descriptors"].shape[0]))
    index = _DescriptorIndex(db["descriptors"], metric=retrieval_metric, use_gpu=bool(faiss_gpu))
    query_indices = list(spec.query_indices[: min(int(timed_queries), len(spec.query_indices))])
    if not query_indices:
        raise RuntimeError(f"No runtime queries for {dataset_name} {spec.sequence_name}.")

    frontend_times: List[float] = []
    retrieval_times: List[float] = []
    verification_times: List[float] = []
    e2e_times: List[float] = []
    selected_ranks: List[int] = []
    selected_by_rerank: List[int] = []
    selected_inliers: List[int] = []
    verified_counts: List[int] = []

    prefer_low = str(retrieval_metric).lower() == "l2"
    meters_per_pixel = float(getattr(db_bank.dataset.meta, "meters_per_pixel", 0.4))

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device=device)

    def merge_tta(scores: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        merged: dict[int, float] = {}
        for row_scores, row_indices in zip(scores, indices):
            for score, index_value in zip(row_scores, row_indices):
                index_value = int(index_value)
                score = float(score)
                if index_value < 0:
                    continue
                if index_value not in merged or (score < merged[index_value] if prefer_low else score > merged[index_value]):
                    merged[index_value] = score
        ordered = sorted(merged.items(), key=lambda item: item[1], reverse=not prefer_low)[:retrieve_k]
        while len(ordered) < retrieve_k:
            fill_index = int(ordered[-1][0]) if ordered else 0
            fill_score = float("inf") if prefer_low else float("-inf")
            ordered.append((fill_index, fill_score))
        return (
            np.asarray([score for _, score in ordered], dtype=np.float32),
            np.asarray([index_value for index_value, _ in ordered], dtype=np.int64),
        )

    def forward_query_tta(query_image: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            if not use_query_tta:
                desc = adapter.forward_global(query_image)
                return descriptors_to_numpy(desc)
            if hasattr(adapter, "forward_global_tta"):
                desc = adapter.forward_global_tta(query_image)
                if desc.ndim == 3:
                    desc = desc[0]
                return descriptors_to_numpy(desc)
            per_rotation = [adapter.forward_global(query_image)]
            for rotation_k in (1, 2, 3):
                rotated = torch.rot90(query_image, k=int(rotation_k), dims=(-2, -1))
                per_rotation.append(adapter.forward_global(rotated))
            desc = torch.cat(per_rotation, dim=0)
            return descriptors_to_numpy(desc)

    total_iters = int(warmup) + len(query_indices)
    for step in range(total_iters):
        query_index = int(query_indices[step % len(query_indices)])
        query_image_cpu = query_bank.dataset[query_index]["image"]
        query_image = query_image_cpu.unsqueeze(0).to(device, non_blocking=True)

        sync()
        start = time.perf_counter()
        query_tta = forward_query_tta(query_image)
        sync()
        frontend_ms = (time.perf_counter() - start) * 1000.0

        sync()
        start = time.perf_counter()
        raw_scores, raw_indices = index.search(query_tta, topk=search_k)
        _, candidate_local_indices = merge_tta(raw_scores, raw_indices)
        sync()
        retrieval_ms = (time.perf_counter() - start) * 1000.0

        verify_local_indices = candidate_local_indices[:verify_v]
        candidate_db_indices = [int(db["indices"][int(local_index)]) for local_index in verify_local_indices]
        candidate_images_cpu = torch.stack(
            [db_bank.dataset[int(db_index)]["image"] for db_index in candidate_db_indices],
            dim=0,
        )

        sync()
        start = time.perf_counter()
        with torch.no_grad():
            query_local_one = adapter.forward_local_features(query_image)
            candidate_local = adapter.forward_local_features(candidate_images_cpu.to(device, non_blocking=True))
            query_images = query_image_cpu.unsqueeze(0).repeat(len(candidate_db_indices), 1, 1, 1)
            query_local = query_local_one.repeat(len(candidate_db_indices), 1, 1, 1)
            results = backend.solve_batch(
                query_images=query_images,
                candidate_images=candidate_images_cpu,
                query_local=query_local,
                candidate_local=candidate_local,
                meters_per_pixel=meters_per_pixel,
            )
        sync()
        verification_ms = (time.perf_counter() - start) * 1000.0

        candidates = [
            {"rank": int(rank), "result": result}
            for rank, result in enumerate(results)
        ]
        selected = _select_reranked_candidate(candidates, rerank_strong_min_inliers)
        e2e_ms = float(frontend_ms + retrieval_ms + verification_ms)

        if step >= int(warmup):
            frontend_times.append(frontend_ms)
            retrieval_times.append(retrieval_ms)
            verification_times.append(verification_ms)
            e2e_times.append(e2e_ms)
            verified_counts.append(int(len(candidate_db_indices)))
            if selected is None:
                selected_ranks.append(0)
                selected_by_rerank.append(0)
                selected_inliers.append(0)
            else:
                rank = int(selected["rank"])
                result = selected.get("result", PairResult.empty())
                selected_ranks.append(rank)
                selected_by_rerank.append(int(rank != 0))
                selected_inliers.append(int(getattr(result, "num_inliers", 0)))

    payload = {
        "dataset": str(dataset_name),
        "runtime_sequence": spec.sequence_name,
        "database": spec.database_sequence_name,
        "protocol": "hybrid_usvloc_tta_topk_bevplacepp_topv_rerank",
        "rerank_top_k": int(retrieve_k),
        "rerank_top_v": int(verify_v),
        "rerank_strong_min_inliers": int(rerank_strong_min_inliers),
        "query_tta_enabled": int(use_query_tta),
        "FrontendFeatureTime": summarize_runtime(frontend_times),
        "RetrievalTime": summarize_runtime(retrieval_times),
        "VerificationRerankTime": summarize_runtime(verification_times),
        "EndToEndTime": summarize_runtime(e2e_times),
        "timed_queries": int(len(query_indices)),
        "warmup": int(warmup),
        "mean_verified_candidates": mean_or_zero(verified_counts),
        "mean_selected_rank": mean_or_zero(selected_ranks),
        "selected_by_rerank_rate": mean_or_zero(selected_by_rerank),
        "mean_selected_inliers": mean_or_zero(selected_inliers),
        "notes": (
            "Strict hybrid runtime: FrontendFeatureTime is the query descriptor stage "
            "(4-rotation TTA when query_tta_enabled=1, single forward otherwise); "
            "RetrievalTime is Top-K search and TTA candidate merge; VerificationRerankTime includes "
            "BEVPlace++ query/candidate local feature extraction for Top-V candidates, sparse RANSAC, "
            "and inlier-based geometric reranking."
        ),
    }
    save_json(output_dir / "paper_runtime.json", payload)
    save_tsv(
        output_dir / "paper_runtime.tsv",
        [
            {"segment": name, **stats}
            for name, stats in payload.items()
            if isinstance(stats, dict) and {"mean", "std", "p50", "p90", "fps"}.issubset(stats.keys())
        ],
    )
    return payload


def evaluate_backend_bundle(
    adapter,
    backend: SparseRansacBackend,
    datasets: Sequence[str],
    processed_root: str | Path,
    output_dir: str | Path,
    device: torch.device,
    metadata: Dict,
    image_size: int = 201,
    eval_batch_size: int = 64,
    pair_batch_size: int = 2,
    num_workers: int = 0,
    retrieval_metric: str = "l2",
    positive_radius_m: float = 5.0,
    negative_radius_m: float = 15.0,
    success_translation_m: float = 2.0,
    success_rotation_deg: float = 5.0,
    kitti_loader_mode: str = "bevplace2_eval_gray3",
    faiss_gpu: bool = True,
    include_runtime: bool = True,
    runtime_warmup: int = 10,
    runtime_timed_queries: int = 50,
    sequence_names: Sequence[str] | None = None,
    max_sequences: int | None = None,
    max_pairs_per_sequence: int | None = None,
    rerank_top_k: int = 1,
    rerank_top_v: int = 1,
    loop_rerank_top_k: int | None = None,
    loop_rerank_top_v: int | None = None,
    global_rerank_top_k: int | None = None,
    global_rerank_top_v: int | None = None,
    rerank_strong_min_inliers: int = 8,
    run_loop: bool = True,
    run_global_loc: bool = True,
) -> Dict:
    output_dir = ensure_dir(output_dir)
    loop_top_k = int(rerank_top_k if loop_rerank_top_k is None else loop_rerank_top_k)
    loop_top_v = int(rerank_top_v if loop_rerank_top_v is None else loop_rerank_top_v)
    global_top_k = int(rerank_top_k if global_rerank_top_k is None else global_rerank_top_k)
    global_top_v = int(rerank_top_v if global_rerank_top_v is None else global_rerank_top_v)
    payload: Dict[str, object] = {
        "metadata": dict(metadata),
        "datasets": [str(dataset) for dataset in datasets],
        "backend": {
            "type": "SparseRansacBackend",
            "score_mode": backend.score_mode,
            "max_keypoints": backend.max_keypoints,
            "max_correspondences": backend.max_correspondences,
            "ransac_iterations": backend.ransac_iterations,
            "ransac_threshold_m": backend.ransac_threshold_m,
            "num_threads": backend.num_threads,
            "loop_rerank_top_k": int(loop_top_k),
            "loop_rerank_top_v": int(loop_top_v),
            "global_rerank_top_k": int(global_top_k),
            "global_rerank_top_v": int(global_top_v),
            "rerank_strong_min_inliers": int(rerank_strong_min_inliers),
        },
    }
    for dataset_name in datasets:
        shared_cache: Dict[str, FeatureBank] = {}
        dataset_dir = ensure_dir(output_dir / str(dataset_name).lower())
        dataset_payload: Dict[str, object] = {}
        if bool(run_loop):
            dataset_payload["loop"] = evaluate_loop(
                adapter,
                backend,
                dataset_name=dataset_name,
                processed_root=processed_root,
                output_dir=dataset_dir / "loop",
                device=device,
                image_size=image_size,
                eval_batch_size=eval_batch_size,
                pair_batch_size=pair_batch_size,
                num_workers=num_workers,
                retrieval_metric=retrieval_metric,
                positive_radius_m=positive_radius_m,
                negative_radius_m=negative_radius_m,
                kitti_loader_mode=kitti_loader_mode,
                faiss_gpu=faiss_gpu,
                cache=shared_cache,
                sequence_names=sequence_names,
                max_sequences=max_sequences,
                max_pairs_per_sequence=max_pairs_per_sequence,
                rerank_top_k=loop_top_k,
                rerank_top_v=loop_top_v,
                rerank_strong_min_inliers=rerank_strong_min_inliers,
            )
        if bool(run_global_loc):
            dataset_payload["global_loc"] = evaluate_global_loc(
                adapter,
                backend,
                dataset_name=dataset_name,
                processed_root=processed_root,
                output_dir=dataset_dir / "global_loc",
                device=device,
                image_size=image_size,
                eval_batch_size=eval_batch_size,
                pair_batch_size=pair_batch_size,
                num_workers=num_workers,
                retrieval_metric=retrieval_metric,
                positive_radius_m=positive_radius_m,
                success_translation_m=success_translation_m,
                success_rotation_deg=success_rotation_deg,
                kitti_loader_mode=kitti_loader_mode,
                faiss_gpu=faiss_gpu,
                cache=shared_cache,
                sequence_names=sequence_names,
                max_sequences=max_sequences,
                max_pairs_per_sequence=max_pairs_per_sequence,
                rerank_top_k=global_top_k,
                rerank_top_v=global_top_v,
                rerank_strong_min_inliers=rerank_strong_min_inliers,
            )
        if include_runtime:
            if str(getattr(adapter, "name", "")).lower() == "usvloc_bevplacepp_hybrid" and int(global_top_v) > 1:
                dataset_payload["runtime"] = benchmark_hybrid_rerank_runtime(
                    adapter,
                    backend,
                    dataset_name=dataset_name,
                    processed_root=processed_root,
                    output_dir=dataset_dir / "runtime",
                    device=device,
                    image_size=image_size,
                    eval_batch_size=eval_batch_size,
                    pair_batch_size=pair_batch_size,
                    num_workers=num_workers,
                    retrieval_metric=retrieval_metric,
                    positive_radius_m=positive_radius_m,
                    kitti_loader_mode=kitti_loader_mode,
                    faiss_gpu=faiss_gpu,
                    warmup=runtime_warmup,
                    timed_queries=runtime_timed_queries,
                    rerank_top_k=global_top_k,
                    rerank_top_v=global_top_v,
                    rerank_strong_min_inliers=rerank_strong_min_inliers,
                    cache=shared_cache,
                    sequence_names=sequence_names,
                )
            else:
                dataset_payload["runtime"] = benchmark_runtime(
                    adapter,
                    backend,
                    dataset_name=dataset_name,
                    processed_root=processed_root,
                    output_dir=dataset_dir / "runtime",
                    device=device,
                    image_size=image_size,
                    eval_batch_size=eval_batch_size,
                    num_workers=num_workers,
                    retrieval_metric=retrieval_metric,
                    positive_radius_m=positive_radius_m,
                    kitti_loader_mode=kitti_loader_mode,
                    faiss_gpu=faiss_gpu,
                    warmup=runtime_warmup,
                    timed_queries=runtime_timed_queries,
                    cache=shared_cache,
                    sequence_names=sequence_names,
                )
        payload[str(dataset_name).lower()] = dataset_payload
        save_json(output_dir / "backend_bundle_summary.json", payload)
    return payload
