from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from usvloc.data.common import ProcessedSequenceDataset
from usvloc.evaluation.retrieval import search_topk
from usvloc.io import ensure_dir, save_json, save_tsv

from .evaluator import (
    RetrievalSpec,
    _build_specs,
    _descriptor_pr_score,
    _filter_specs_by_sequence_names,
    _min_distances_to_database,
    _pose2d_from_xyyaw,
    _pred_pose2d_from_relative,
    _to_numpy_position,
    _with_mean_row,
)
from .frontends import descriptors_to_numpy
from .metrics import (
    area_under_pr,
    best_f1_and_threshold,
    max_recall_at_precision,
    mean_or_zero,
    precision_recall_curve,
    summarize_runtime,
)
from .polar_ransac_backend import (
    TOP_K_RETRIEVAL,
    TOP_V_VERIFY,
    TTA_ROTATIONS,
    PolarCandidate,
    PolarRansacBackend,
    angular_xcorr_profiles,
    polar_profile_from_features,
    rerank_score,
)
from .types import PairResult


@dataclass
class PolarFeatureBank:
    sequence_name: str
    sequence_dir: str
    descriptors: np.ndarray
    polar_profiles: np.ndarray
    position: np.ndarray
    xy: np.ndarray
    yaw: np.ndarray
    indices: np.ndarray
    dataset: ProcessedSequenceDataset
    tta_descriptors: np.ndarray | None = None


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


def _extract_polar_feature_bank(
    adapter,
    sequence_dir: str | Path,
    image_size: int,
    batch_size: int,
    device: torch.device,
    cache: Dict[tuple[str, bool, str], PolarFeatureBank] | None,
    kitti_loader_mode: str,
    num_workers: int,
    include_tta: bool,
    selected_indices: Sequence[int] | np.ndarray | None = None,
) -> PolarFeatureBank:
    if selected_indices is None:
        selection_token = "all"
        selected_list: list[int] | None = None
    else:
        selected_array = np.asarray(selected_indices, dtype=np.int64)
        selected_list = [int(v) for v in selected_array.tolist()]
        selection_token = f"{len(selected_list)}:{selected_list[0] if selected_list else -1}:{selected_list[-1] if selected_list else -1}:{int(selected_array.sum()) if selected_array.size else 0}"
    key = (str(Path(sequence_dir).resolve()), bool(include_tta), selection_token)
    if cache is not None and key in cache:
        return cache[key]

    dataset = ProcessedSequenceDataset(
        sequence_dir,
        image_size=int(image_size),
        split_tags=None,
        kitti_loader_mode=str(kitti_loader_mode),
    )
    loader_dataset = dataset if selected_list is None else Subset(dataset, selected_list)
    loader = DataLoader(loader_dataset, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers))
    descriptors: List[np.ndarray] = []
    polar_profiles: List[np.ndarray] = []
    tta_descriptors: List[np.ndarray] = []
    positions: List[np.ndarray] = []
    xys: List[np.ndarray] = []
    yaws: List[np.ndarray] = []
    frame_indices: List[np.ndarray] = []
    adapter.model.eval()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            desc, polar, _ = adapter.forward_backend_features(images, output_size=tuple(int(v) for v in images.shape[-2:]))
            descriptors.append(descriptors_to_numpy(desc))
            profile = polar_profile_from_features(polar).detach().cpu().numpy().astype(np.float16, copy=False)
            polar_profiles.append(profile)
            if include_tta:
                per_rotation: List[np.ndarray] = [descriptors_to_numpy(desc)]
                for rotation_k in TTA_ROTATIONS[1:]:
                    rotated = torch.rot90(images, k=int(rotation_k), dims=(-2, -1))
                    per_rotation.append(descriptors_to_numpy(adapter.forward_global(rotated)))
                tta_descriptors.append(np.stack(per_rotation, axis=1).astype(np.float32, copy=False))
            position, xy, yaw = _to_numpy_position(batch, dataset.meta.dataset_name)
            positions.append(position)
            xys.append(xy)
            yaws.append(yaw)
            frame_indices.append(batch["index"].detach().cpu().numpy().astype(np.int64))

    bank = PolarFeatureBank(
        sequence_name=str(dataset.meta.sequence_name),
        sequence_dir=key[0],
        descriptors=np.concatenate(descriptors, axis=0).astype(np.float32, copy=False),
        polar_profiles=np.concatenate(polar_profiles, axis=0).astype(np.float16, copy=False),
        position=np.concatenate(positions, axis=0).astype(np.float32, copy=False),
        xy=np.concatenate(xys, axis=0).astype(np.float32, copy=False),
        yaw=np.concatenate(yaws, axis=0).astype(np.float32, copy=False),
        indices=np.concatenate(frame_indices, axis=0).astype(np.int64, copy=False),
        dataset=dataset,
        tta_descriptors=np.concatenate(tta_descriptors, axis=0).astype(np.float32, copy=False) if include_tta else None,
    )
    if cache is not None:
        cache[key] = bank
    return bank


def _subset_bank(bank: PolarFeatureBank, indices: Iterable[int] | None = None) -> Dict[str, np.ndarray]:
    if indices is None:
        indices = np.arange(len(bank.indices), dtype=np.int64)
    else:
        indices = np.asarray(list(indices), dtype=np.int64)
    subset = {
        "descriptors": bank.descriptors[indices],
        "polar_profiles": bank.polar_profiles[indices],
        "position": bank.position[indices],
        "xy": bank.xy[indices],
        "yaw": bank.yaw[indices],
        "indices": indices,
    }
    if bank.tta_descriptors is not None:
        subset["tta_descriptors"] = bank.tta_descriptors[indices]
    return subset


def _search_topk_tta(
    db_descs: np.ndarray,
    query_descs: np.ndarray,
    query_tta_descs: np.ndarray | None,
    topk: int,
    metric: str,
    use_gpu: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if query_tta_descs is None:
        scores, indices = search_topk(db_descs, query_descs, topk=int(topk), metric=metric, use_gpu=bool(use_gpu))
        return scores, indices, None
    num_queries, num_rot, dim = query_tta_descs.shape
    flat = query_tta_descs.reshape(num_queries * num_rot, dim)
    search_k = min(max(int(topk) * 2, int(topk)), int(db_descs.shape[0]))
    flat_scores, flat_indices = search_topk(db_descs, flat, topk=search_k, metric=metric, use_gpu=bool(use_gpu))
    scores = flat_scores.reshape(num_queries, num_rot, search_k)
    indices = flat_indices.reshape(num_queries, num_rot, search_k)
    out_scores = np.empty((num_queries, int(topk)), dtype=np.float32)
    out_indices = np.empty((num_queries, int(topk)), dtype=np.int64)
    out_rotations = np.empty((num_queries, int(topk)), dtype=np.int64)
    prefer_low = str(metric).lower() == "l2"
    for qi in range(num_queries):
        merged: dict[int, tuple[float, int]] = {}
        for ri in range(num_rot):
            for score, idx in zip(scores[qi, ri], indices[qi, ri]):
                idx = int(idx)
                score = float(score)
                if idx < 0:
                    continue
                if idx not in merged or (score < merged[idx][0] if prefer_low else score > merged[idx][0]):
                    merged[idx] = (score, int(ri))
        ordered = sorted(merged.items(), key=lambda item: item[1][0], reverse=not prefer_low)[: int(topk)]
        while len(ordered) < int(topk):
            ordered.append((int(ordered[-1][0]) if ordered else 0, (float("inf") if prefer_low else float("-inf"), -1)))
        out_indices[qi] = np.asarray([idx for idx, _ in ordered], dtype=np.int64)
        out_scores[qi] = np.asarray([score for _, (score, _) in ordered], dtype=np.float32)
        out_rotations[qi] = np.asarray([rot for _, (_, rot) in ordered], dtype=np.int64)
    return out_scores, out_indices, out_rotations


def _load_images(dataset: ProcessedSequenceDataset, indices: Sequence[int]) -> torch.Tensor:
    return torch.stack([dataset[int(index)]["image"] for index in indices], dim=0)


def _verify_query(
    adapter,
    backend: PolarRansacBackend,
    query_bank: PolarFeatureBank,
    db_bank: PolarFeatureBank,
    query_index: int,
    candidate_orig_indices: Sequence[int],
    candidates,
    device: torch.device,
) -> tuple[PairResult, object | None]:
    query_image_cpu = query_bank.dataset[int(query_index)]["image"]
    candidate_images_cpu = _load_images(db_bank.dataset, candidate_orig_indices)
    images = torch.cat([query_image_cpu.unsqueeze(0), candidate_images_cpu], dim=0).to(device, non_blocking=True)
    with torch.no_grad():
        _, _, local = adapter.forward_backend_features(images, output_size=tuple(int(v) for v in images.shape[-2:]))
    output = backend.verify_top_candidates(
        query_image=query_image_cpu,
        candidate_images=candidate_images_cpu,
        query_local=local[0],
        candidate_local=local[1:],
        candidates=candidates,
        meters_per_pixel=float(query_bank.dataset.meta.meters_per_pixel),
    )
    return output.result, output.candidate


def _paper_top1_candidate(
    query_profile: np.ndarray,
    candidate_profiles: np.ndarray,
    top1_local_index: int,
    top1_distance: float,
) -> PolarCandidate:
    theta, peak_ratio, _ = angular_xcorr_profiles(
        np.asarray(query_profile, dtype=np.float32),
        candidate_profiles[int(top1_local_index)].astype(np.float32, copy=False),
    )
    return PolarCandidate(
        local_index=int(top1_local_index),
        retrieval_rank=0,
        retrieval_dist=float(top1_distance),
        theta_rad=float(theta),
        peak_ratio=float(peak_ratio),
        rerank_score=float(rerank_score(float(top1_distance), float(peak_ratio))),
    )


def _prepare_sequence(
    adapter,
    spec: RetrievalSpec,
    cache: Dict[tuple[str, bool, str], PolarFeatureBank],
    image_size: int,
    eval_batch_size: int,
    device: torch.device,
    kitti_loader_mode: str,
    num_workers: int,
) -> tuple[PolarFeatureBank, PolarFeatureBank, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    db_bank = _extract_polar_feature_bank(
        adapter,
        spec.database_sequence_dir,
        image_size=image_size,
        batch_size=eval_batch_size,
        device=device,
        cache=cache,
        kitti_loader_mode=kitti_loader_mode,
        num_workers=num_workers,
        include_tta=False,
        selected_indices=spec.database_indices,
    )
    query_bank = _extract_polar_feature_bank(
        adapter,
        spec.query_sequence_dir,
        image_size=image_size,
        batch_size=max(1, eval_batch_size // 2),
        device=device,
        cache=cache,
        kitti_loader_mode=kitti_loader_mode,
        num_workers=num_workers,
        include_tta=True,
        selected_indices=spec.query_indices,
    )
    return db_bank, query_bank, _subset_bank(db_bank), _subset_bank(query_bank)


def evaluate_polar_loop(
    adapter,
    backend: PolarRansacBackend,
    dataset_name: str,
    processed_root: str | Path,
    output_dir: str | Path,
    device: torch.device,
    image_size: int = 201,
    eval_batch_size: int = 64,
    num_workers: int = 0,
    retrieval_metric: str = "l2",
    positive_radius_m: float = 5.0,
    negative_radius_m: float = 15.0,
    kitti_loader_mode: str = "bevplace2_eval_gray3",
    faiss_gpu: bool = True,
    cache: Dict[tuple[str, bool, str], PolarFeatureBank] | None = None,
    sequence_names: Sequence[str] | None = None,
    max_sequences: int | None = None,
    max_pairs_per_sequence: int | None = None,
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
        print(f"[PolarLoop] {dataset_name} seq={spec.sequence_name} db={spec.database_sequence_name} descriptors+polar...", flush=True)
        db_bank, query_bank, db, query = _prepare_sequence(
            adapter, spec, cache, image_size, eval_batch_size, device, kitti_loader_mode, num_workers
        )
        retrieval_scores, predictions, tta_rotations = _search_topk_tta(
            db["descriptors"],
            query["descriptors"],
            query.get("tta_descriptors", None),
            topk=TOP_K_RETRIEVAL,
            metric=retrieval_metric,
            use_gpu=bool(faiss_gpu),
        )
        debug_limit = int(os.environ.get("USVLOC_POLAR_DEBUG_LIMIT", "0") or 0)
        db_profiles_f32 = db["polar_profiles"].astype(np.float32, copy=False)
        min_gt_distances = _min_distances_to_database(query["position"], db["position"])
        has_positive = min_gt_distances < float(positive_radius_m)
        total_positive_queries = int(np.sum(has_positive[: len(query["indices"])]))
        query_lookup = {int(idx): pos for pos, idx in enumerate(query["indices"])}
        db_lookup = {int(idx): pos for pos, idx in enumerate(db["indices"])}
        labels: List[int] = []
        scores: List[float] = []
        details: List[Dict[str, object]] = []
        max_queries = len(query["indices"])
        if max_pairs_per_sequence is not None and int(max_pairs_per_sequence) > 0:
            max_queries = min(max_queries, int(max_pairs_per_sequence))

        for qi in range(max_queries):
            top1_local = int(predictions[qi, 0])
            top1_score = float(retrieval_scores[qi, 0])
            if qi < debug_limit:
                selected_tta_rot_k = int(tta_rotations[qi, 0]) if tta_rotations is not None else 0
                print(
                    f"[debug_tta] loop_query={qi} selected_tta_rot_k={selected_tta_rot_k} "
                    "geometry_query_rot_k=0",
                    flush=True,
                )
            paper_candidate = _paper_top1_candidate(
                query_profile=query["polar_profiles"][qi].astype(np.float32),
                candidate_profiles=db_profiles_f32,
                top1_local_index=top1_local,
                top1_distance=top1_score,
            )
            candidate_orig = [int(db["indices"][top1_local])]
            result, best_candidate = _verify_query(
                adapter, backend, query_bank, db_bank, int(query["indices"][qi]), candidate_orig, [paper_candidate], device
            )
            best_candidate = best_candidate or paper_candidate
            best_local = top1_local
            dist = float(np.linalg.norm(db["position"][best_local] - query["position"][qi]))
            label = int(dist < float(positive_radius_m))
            pr_score = _descriptor_pr_score(top1_score, retrieval_metric)
            q_idx = int(query["indices"][qi])
            d_idx = int(db["indices"][best_local])
            row: Dict[str, object] = {
                "dataset": str(dataset_name),
                "sequence": spec.sequence_name,
                "database": spec.database_sequence_name,
                "query_index": q_idx,
                "db_index": d_idx,
                "label": int(label),
                "has_positive": int(has_positive[qi]),
                "nearest_gt_db_distance_m": float(min_gt_distances[qi]),
                "retrieval_distance_m": dist,
                "descriptor_raw_score": float(top1_score),
                "score": float(pr_score),
                "backend_score": float(result.score),
                "pose_valid": int(result.pose_valid),
                "num_inliers": int(result.num_inliers),
                "num_matches": int(result.num_matches),
                "inlier_mean_residual_m": float(result.inlier_mean_residual_m) if np.isfinite(result.inlier_mean_residual_m) else None,
            }
            row["retrieval_rank"] = int(best_candidate.retrieval_rank)
            row["peak_ratio"] = float(best_candidate.peak_ratio)
            row["theta_acc_deg"] = float(np.degrees(best_candidate.theta_rad))
            if result.pose_valid:
                q_local = query_lookup[q_idx]
                d_local = db_lookup[d_idx]
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
            details.append(row)
            labels.append(label)
            scores.append(float(pr_score))
            if (qi + 1) == max_queries or (qi + 1) == 1 or (qi + 1) % 100 == 0:
                print(f"[PolarBackend] loop query {qi + 1}/{max_queries}", flush=True)

        if max_pairs_per_sequence is not None and int(max_pairs_per_sequence) > 0:
            total_positive_queries = int(sum(int(row["has_positive"]) for row in details))

        print(
            f"[PolarLoop] {dataset_name} seq={spec.sequence_name} pairs={len(details)} "
            f"top1_tp={int(sum(labels))} gt_positive_queries={total_positive_queries}",
            flush=True,
        )
        labels_np = np.asarray(labels, dtype=np.int64)
        scores_np = np.asarray(scores, dtype=np.float32)
        curve = precision_recall_curve(scores_np, labels_np, total_positives=total_positive_queries)
        max_f1, threshold = best_f1_and_threshold(curve["precision"], curve["recall"], curve["thresholds"])
        true_positive_pose_rows = [
            row
            for row in details
            if int(row["label"]) == 1
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
            "num_positive_queries": int(total_positive_queries),
            "num_true_positive_pose_queries": int(len(true_positive_pose_rows)),
        }
        per_sequence.append(summary)
        all_pairs.extend(details)
        print(
            f"[PolarLoop] {dataset_name} seq={spec.sequence_name} AP={summary['AP']:.4f} "
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
            "notes": "BEVPlace2-main aligned protocol: KITTI positives use raw 3D pose distance [x,y,z]; KITTI pose error uses BEV ground-plane [x,z]+yaw. Loop PR/AP/MaxF1/MaxRecall@100P threshold the descriptor Top-1 distance; e_t/e_r are averaged over true-positive Top-1 queries with valid pose estimates. USVLoc descriptors use 4-rotation TTA before selecting Top-1.",
    }
    save_json(output_dir / "paper_loop_v4.json", payload)
    save_tsv(output_dir / "paper_loop_v4.tsv", _with_mean_row(per_sequence, ["AP", "MaxF1", "MaxRecall@100P", "e_t_m", "e_r_deg"]))
    save_tsv(output_dir / "paper_loop_pairs_v4.tsv", all_pairs)
    return payload


def evaluate_polar_global_loc(
    adapter,
    backend: PolarRansacBackend,
    dataset_name: str,
    processed_root: str | Path,
    output_dir: str | Path,
    device: torch.device,
    image_size: int = 201,
    eval_batch_size: int = 64,
    num_workers: int = 0,
    retrieval_metric: str = "l2",
    positive_radius_m: float = 5.0,
    success_translation_m: float = 2.0,
    success_rotation_deg: float = 5.0,
    kitti_loader_mode: str = "bevplace2_eval_gray3",
    faiss_gpu: bool = True,
    cache: Dict[tuple[str, bool, str], PolarFeatureBank] | None = None,
    sequence_names: Sequence[str] | None = None,
    max_sequences: int | None = None,
    max_pairs_per_sequence: int | None = None,
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
        print(f"[PolarGlobalLoc] {dataset_name} seq={spec.sequence_name} db={spec.database_sequence_name} descriptors+polar...", flush=True)
        db_bank, query_bank, db, query = _prepare_sequence(
            adapter, spec, cache, image_size, eval_batch_size, device, kitti_loader_mode, num_workers
        )
        retrieval_scores, predictions, tta_rotations = _search_topk_tta(
            db["descriptors"],
            query["descriptors"],
            query.get("tta_descriptors", None),
            topk=TOP_K_RETRIEVAL,
            metric=retrieval_metric,
            use_gpu=bool(faiss_gpu),
        )
        debug_limit = int(os.environ.get("USVLOC_POLAR_DEBUG_LIMIT", "0") or 0)
        db_profiles_f32 = db["polar_profiles"].astype(np.float32, copy=False)
        query_lookup = {int(idx): pos for pos, idx in enumerate(query["indices"])}
        db_lookup = {int(idx): pos for pos, idx in enumerate(db["indices"])}
        details: List[Dict[str, object]] = []
        max_queries = len(query["indices"])
        if max_pairs_per_sequence is not None and int(max_pairs_per_sequence) > 0:
            max_queries = min(max_queries, int(max_pairs_per_sequence))

        for qi in range(max_queries):
            distances = np.linalg.norm(db["position"] - query["position"][qi : qi + 1], axis=1)
            positives = np.where(distances < float(positive_radius_m))[0]
            if positives.size == 0:
                continue
            top1_local = int(predictions[qi, 0])
            top1_score = float(retrieval_scores[qi, 0])
            if qi < debug_limit:
                selected_tta_rot_k = int(tta_rotations[qi, 0]) if tta_rotations is not None else 0
                print(
                    f"[debug_tta] global_query={qi} selected_tta_rot_k={selected_tta_rot_k} "
                    "geometry_query_rot_k=0",
                    flush=True,
                )
            paper_candidate = _paper_top1_candidate(
                query_profile=query["polar_profiles"][qi].astype(np.float32),
                candidate_profiles=db_profiles_f32,
                top1_local_index=top1_local,
                top1_distance=top1_score,
            )
            candidate_orig = [int(db["indices"][top1_local])]
            result, best_candidate = _verify_query(
                adapter, backend, query_bank, db_bank, int(query["indices"][qi]), candidate_orig, [paper_candidate], device
            )
            best_candidate = best_candidate or paper_candidate
            best_local = top1_local
            retrieval_hit = int(best_local in set(int(v) for v in positives.tolist()))
            q_idx = int(query["indices"][qi])
            d_idx = int(db["indices"][best_local])
            row: Dict[str, object] = {
                "dataset": str(dataset_name),
                "sequence": spec.sequence_name,
                "database": spec.database_sequence_name,
                "query_index": q_idx,
                "db_index": d_idx,
                "descriptor_score": float(top1_score),
                "nearest_gt_db_distance_m": float(np.min(distances)) if distances.size else float("inf"),
                "retrieval_distance_m": float(distances[best_local]),
                "has_positive": 1,
                "retrieval_hit": int(retrieval_hit),
                "backend_score": float(result.score),
                "pose_valid": int(result.pose_valid),
                "num_inliers": int(result.num_inliers),
                "num_matches": int(result.num_matches),
                "e_t_m": None,
                "e_r_deg": None,
                "success": 0,
            }
            row["retrieval_rank"] = int(best_candidate.retrieval_rank)
            row["peak_ratio"] = float(best_candidate.peak_ratio)
            row["theta_acc_deg"] = float(np.degrees(best_candidate.theta_rad))
            if result.pose_valid:
                q_local = query_lookup[q_idx]
                d_local = db_lookup[d_idx]
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
            details.append(row)
            if (qi + 1) == max_queries or (qi + 1) == 1 or (qi + 1) % 100 == 0:
                print(f"[PolarBackend] global query {qi + 1}/{max_queries}", flush=True)

        success_rows = [row for row in details if int(row["success"]) == 1]
        summary = {
            "dataset": str(dataset_name),
            "sequence": spec.sequence_name,
            "database": spec.database_sequence_name,
            "Recall@1": float(sum(int(row["retrieval_hit"]) for row in details)) / float(max(len(details), 1)),
            "SuccessRate": float(len(success_rows)) / float(max(len(details), 1)),
            "MeanTransErr": mean_or_zero([float(row["e_t_m"]) for row in success_rows if row["e_t_m"] is not None]),
            "MeanRotErr": mean_or_zero([float(row["e_r_deg"]) for row in success_rows if row["e_r_deg"] is not None]),
            "AllPositiveQueries": int(len(details)),
            "SuccessfulQueries": int(len(success_rows)),
        }
        per_sequence.append(summary)
        all_pairs.extend(details)
        print(
            f"[PolarGlobalLoc] {dataset_name} seq={spec.sequence_name} R@1={summary['Recall@1']:.4f} "
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
            "notes": "BEVPlace2-main aligned protocol: KITTI positives use raw 3D pose distance [x,y,z]; KITTI pose error uses BEV ground-plane [x,z]+yaw. Descriptor Top-1 is used for Recall@1 and pose estimation; SR uses the (2 m, 5 deg) threshold; e_t/e_r are averaged over successful localizations. USVLoc descriptors use 4-rotation TTA before selecting Top-1.",
    }
    save_json(output_dir / "paper_global_loc_v4.json", payload)
    save_tsv(output_dir / "paper_global_loc_v4.tsv", _with_mean_row(per_sequence, ["Recall@1", "SuccessRate", "MeanTransErr", "MeanRotErr"]))
    save_tsv(output_dir / "paper_global_loc_pairs_v4.tsv", all_pairs)
    return payload


def benchmark_polar_runtime(
    adapter,
    backend: PolarRansacBackend,
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
    cache: Dict[tuple[str, bool, str], PolarFeatureBank] | None = None,
) -> Dict:
    output_dir = ensure_dir(output_dir)
    spec = _build_specs(dataset_name, processed_root, positive_radius_m=positive_radius_m)[-1 if str(dataset_name).lower() == "kitti" else 0]
    if cache is None:
        cache = {}
    db_bank = _extract_polar_feature_bank(
        adapter,
        spec.database_sequence_dir,
        image_size=image_size,
        batch_size=eval_batch_size,
        device=device,
        cache=cache,
        kitti_loader_mode=kitti_loader_mode,
        num_workers=num_workers,
        include_tta=False,
        selected_indices=spec.database_indices,
    )
    db = _subset_bank(db_bank)
    db_profiles_f32 = db["polar_profiles"].astype(np.float32, copy=False)
    query_dataset = ProcessedSequenceDataset(
        spec.query_sequence_dir,
        image_size=int(image_size),
        split_tags=None,
        kitti_loader_mode=str(kitti_loader_mode),
    )
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

    total_iters = int(warmup) + len(query_indices)
    for step in range(total_iters):
        query_index = int(query_indices[step % len(query_indices)])
        query_image_cpu = query_dataset[query_index]["image"]
        query_image = query_image_cpu.unsqueeze(0).to(device, non_blocking=True)

        sync()
        start = time.perf_counter()
        with torch.no_grad():
            query_desc, query_polar, query_local = adapter.forward_backend_features(query_image, output_size=tuple(query_image.shape[-2:]))
            tta_descs = [descriptors_to_numpy(query_desc)[0]]
            for rotation_k in TTA_ROTATIONS[1:]:
                rotated = torch.rot90(query_image, k=int(rotation_k), dims=(-2, -1))
                tta_descs.append(descriptors_to_numpy(adapter.forward_global(rotated))[0])
            query_profile = polar_profile_from_features(query_polar)[0].detach().cpu().numpy().astype(np.float32)
        sync()
        frontend_ms = (time.perf_counter() - start) * 1000.0

        query_tta = np.asarray(tta_descs, dtype=np.float32).reshape(1, len(TTA_ROTATIONS), -1)
        sync()
        start = time.perf_counter()
        retrieval_scores, predictions, _ = _search_topk_tta(
            db["descriptors"],
            descriptors_to_numpy(query_desc),
            query_tta,
            topk=TOP_K_RETRIEVAL,
            metric=retrieval_metric,
            use_gpu=bool(faiss_gpu),
        )
        sync()
        retrieval_ms = (time.perf_counter() - start) * 1000.0

        sync()
        start = time.perf_counter()
        reranked = backend.rerank(
            query_profile=query_profile,
            candidate_profiles=db_profiles_f32,
            candidate_indices=predictions[0].tolist(),
            retrieval_dists=retrieval_scores[0].tolist(),
        )
        candidate_orig = [int(db["indices"][candidate.local_index]) for candidate in reranked]
        candidate_images_cpu = _load_images(db_bank.dataset, candidate_orig)
        candidate_images = candidate_images_cpu.to(device, non_blocking=True)
        with torch.no_grad():
            _, _, candidate_local = adapter.forward_backend_features(candidate_images, output_size=tuple(candidate_images.shape[-2:]))
        _ = backend.verify_top_candidates(
            query_image=query_image_cpu,
            candidate_images=candidate_images_cpu,
            query_local=query_local[0],
            candidate_local=candidate_local,
            candidates=reranked,
            meters_per_pixel=float(db_bank.dataset.meta.meters_per_pixel),
        )
        sync()
        verification_ms = (time.perf_counter() - start) * 1000.0
        e2e_ms = frontend_ms + retrieval_ms + verification_ms

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
        "notes": "USVLoc polar runtime includes 4-rotation query descriptor TTA, light ACC reranking, top-5 candidate feature extraction, and BEVPlace2-style 2-point full rigid RANSAC verification.",
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


def evaluate_polar_backend_bundle(
    adapter,
    backend: PolarRansacBackend,
    datasets: Sequence[str],
    processed_root: str | Path,
    output_dir: str | Path,
    device: torch.device,
    metadata: Dict,
    image_size: int = 201,
    eval_batch_size: int = 64,
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
    rerank_top_k: int = TOP_K_RETRIEVAL,
    rerank_top_v: int = TOP_V_VERIFY,
    loop_rerank_top_k: int | None = None,
    loop_rerank_top_v: int | None = None,
    global_rerank_top_k: int | None = None,
    global_rerank_top_v: int | None = None,
    run_loop: bool = True,
    run_global_loc: bool = True,
) -> Dict:
    output_dir = ensure_dir(output_dir)
    payload: Dict[str, object] = {
        "metadata": dict(metadata),
        "datasets": [str(dataset) for dataset in datasets],
        "backend": {
            "type": "PolarRansacBackend",
            "top_k_retrieval": TOP_K_RETRIEVAL,
            "top_v_verify": TOP_V_VERIFY,
            "requested_rerank_top_k": int(rerank_top_k),
            "requested_rerank_top_v": int(rerank_top_v),
            "requested_loop_rerank_top_k": None if loop_rerank_top_k is None else int(loop_rerank_top_k),
            "requested_loop_rerank_top_v": None if loop_rerank_top_v is None else int(loop_rerank_top_v),
            "requested_global_rerank_top_k": None if global_rerank_top_k is None else int(global_rerank_top_k),
            "requested_global_rerank_top_v": None if global_rerank_top_v is None else int(global_rerank_top_v),
            "loop_inlier_threshold": backend.loop_inlier_threshold,
            "loc_inlier_threshold": backend.loc_inlier_threshold,
        },
    }
    for dataset_name in datasets:
        shared_cache: Dict[tuple[str, bool, str], PolarFeatureBank] = {}
        dataset_dir = ensure_dir(output_dir / str(dataset_name).lower())
        dataset_payload: Dict[str, object] = {}
        if bool(run_loop):
            dataset_payload["loop"] = evaluate_polar_loop(
                adapter=adapter,
                backend=backend,
                dataset_name=dataset_name,
                processed_root=processed_root,
                output_dir=dataset_dir / "loop",
                device=device,
                image_size=image_size,
                eval_batch_size=eval_batch_size,
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
            )
        if bool(run_global_loc):
            dataset_payload["global_loc"] = evaluate_polar_global_loc(
                adapter=adapter,
                backend=backend,
                dataset_name=dataset_name,
                processed_root=processed_root,
                output_dir=dataset_dir / "global_loc",
                device=device,
                image_size=image_size,
                eval_batch_size=eval_batch_size,
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
            )
        if include_runtime:
            dataset_payload["runtime"] = benchmark_polar_runtime(
                adapter=adapter,
                backend=backend,
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
            )
        payload[str(dataset_name).lower()] = dataset_payload
        save_json(output_dir / "backend_bundle_summary.json", payload)
    return payload
