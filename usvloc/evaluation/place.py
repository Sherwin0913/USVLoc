from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data import ProcessedSequenceDataset, build_kitti_place_spec, build_nclt_place_specs, build_pohang_place_specs
from ..data.common import resolve_dataset_root
from ..io import ensure_dir, save_json, save_tsv
from .retrieval import compute_recall_at_k, compute_recall_at_one_percent, search_topk


@dataclass
class FeatureBank:
    """Frontend feature cache for one sequence.

    ``descriptors`` are used for global retrieval. ``position/xy/yaw`` are used
    for positive matching, diagnostics, and backend evaluation. ``indices`` keep
    the original frame indices.
    """

    descriptors: np.ndarray
    tta_descriptors: np.ndarray | None
    position: np.ndarray
    xy: np.ndarray
    yaw: np.ndarray
    indices: np.ndarray
    dataset: ProcessedSequenceDataset


def _build_processed_dataset_kwargs(cfg: Dict) -> Dict[str, object]:
    dataset_cfg = cfg.get("dataset", {})
    dataset_name = str(dataset_cfg.get("name", "kitti")).lower()
    if dataset_name != "kitti":
        return {}
    kwargs: Dict[str, object] = {}
    kitti_loader_mode = dataset_cfg.get("kitti_loader_eval_mode", dataset_cfg.get("kitti_loader_mode", None))
    if kitti_loader_mode is not None:
        kwargs["kitti_loader_mode"] = str(kitti_loader_mode)
    else:
        kwargs["kitti_original_like_loader"] = bool(dataset_cfg.get("kitti_original_like_loader", True))
    if dataset_cfg.get("expected_meters_per_pixel", None) is not None:
        kwargs["expected_meters_per_pixel"] = float(dataset_cfg["expected_meters_per_pixel"])
    if dataset_cfg.get("expected_model_input_size", None) is not None:
        kwargs["expected_model_input_size"] = int(dataset_cfg["expected_model_input_size"])
    return kwargs


def _list_kitti_sequences(processed_root: str | Path) -> list[str]:
    kitti_root = resolve_dataset_root(processed_root, "kitti")
    sequences = sorted(path.name for path in kitti_root.iterdir() if path.is_dir())
    if not sequences:
        raise RuntimeError(f"No KITTI sequences found in {kitti_root}")
    return sequences


def _list_nclt_sequences(processed_root: str | Path) -> list[str]:
    nclt_root = resolve_dataset_root(processed_root, "nclt")
    sequences = sorted(path.name for path in nclt_root.iterdir() if path.is_dir())
    if not sequences:
        raise RuntimeError(f"No NCLT sequences found in {nclt_root}")
    return sequences


def _resolve_eval_sequences(cfg: Dict) -> list[str]:
    eval_cfg = cfg.get("evaluation", {})
    dataset_name = str(cfg.get("dataset", {}).get("name", "kitti")).lower()
    sequences = eval_cfg.get("sequences", None)
    if sequences is None:
        if dataset_name == "nclt":
            database_sequence = str(
                eval_cfg.get(
                    "database_sequence",
                    cfg.get("dataset", {}).get("eval_database_sequence", "2012-01-15"),
                )
            )
            return [sequence for sequence in _list_nclt_sequences(cfg["dataset"]["processed_root"]) if sequence != database_sequence]
        return _list_kitti_sequences(cfg["dataset"]["processed_root"])
    if isinstance(sequences, str):
        if sequences.lower() == "all":
            if dataset_name == "nclt":
                database_sequence = str(
                    eval_cfg.get(
                        "database_sequence",
                        cfg.get("dataset", {}).get("eval_database_sequence", "2012-01-15"),
                    )
                )
                return [sequence for sequence in _list_nclt_sequences(cfg["dataset"]["processed_root"]) if sequence != database_sequence]
            return _list_kitti_sequences(cfg["dataset"]["processed_root"])
        return [str(sequences)]
    resolved = [str(sequence) for sequence in sequences]
    if not resolved:
        raise RuntimeError("evaluation.sequences is empty.")
    return resolved


def _get_positive_radius_m(cfg: Dict) -> float:
    eval_cfg = cfg.get("evaluation", {})
    return float(eval_cfg.get("place_positive_radius_m", eval_cfg.get("positive_radius_m", 5.0)))


def _extract_feature_bank(
    model,
    sequence_dir: str | Path,
    image_size: int,
    batch_size: int,
    device: torch.device,
    processed_dataset_kwargs: Dict[str, object],
    cache: dict[str, FeatureBank],
    include_tta: bool = False,
) -> FeatureBank:
    key = f"{Path(sequence_dir).resolve()}|tta={int(bool(include_tta))}"
    if key in cache:
        return cache[key]

    dataset = ProcessedSequenceDataset(
        sequence_dir,
        image_size=image_size,
        split_tags=None,
        **processed_dataset_kwargs,
    )
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    descriptors: list[np.ndarray] = []
    tta_descriptors: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    xys: list[np.ndarray] = []
    yaws: list[np.ndarray] = []
    indices: list[np.ndarray] = []

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            desc_tensor = model.forward_retrieval(images)["global_descriptor"]
            desc = desc_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            descriptors.append(desc)
            if bool(include_tta):
                per_rotation = [desc]
                for rotation_k in (1, 2, 3):
                    rotated = torch.rot90(images, k=int(rotation_k), dims=(-2, -1))
                    rotated_desc = model.forward_retrieval(rotated)["global_descriptor"]
                    per_rotation.append(rotated_desc.detach().cpu().numpy().astype(np.float32, copy=False))
                tta_descriptors.append(np.stack(per_rotation, axis=1).astype(np.float32, copy=False))
            positions.append(
                np.stack(
                    [
                        np.asarray(batch["x_m"], dtype=np.float64),
                        np.asarray(batch["y_m"], dtype=np.float64),
                        np.asarray(batch["z_m"], dtype=np.float64),
                    ],
                    axis=1,
                )
            )
            xys.append(
                np.stack(
                    [
                        np.asarray(batch["x_m"], dtype=np.float64),
                        np.asarray(batch["y_m"], dtype=np.float64),
                    ],
                    axis=1,
                )
            )
            yaws.append(np.asarray(batch["yaw_rad"], dtype=np.float64))
            indices.append(np.asarray(batch["index"], dtype=np.int64))
    if was_training:
        model.train()

    bank = FeatureBank(
        descriptors=np.concatenate(descriptors, axis=0) if descriptors else np.zeros((0, model.global_descriptor_dim), dtype=np.float32),
        tta_descriptors=(
            np.concatenate(tta_descriptors, axis=0).astype(np.float32, copy=False)
            if tta_descriptors
            else None
        ),
        position=np.concatenate(positions, axis=0) if positions else np.zeros((0, 3), dtype=np.float64),
        xy=np.concatenate(xys, axis=0) if xys else np.zeros((0, 2), dtype=np.float64),
        yaw=np.concatenate(yaws, axis=0) if yaws else np.zeros((0,), dtype=np.float64),
        indices=np.concatenate(indices, axis=0) if indices else np.zeros((0,), dtype=np.int64),
        dataset=dataset,
    )
    cache[key] = bank
    return bank


def _get_subset(bank: FeatureBank, indices: np.ndarray) -> Dict[str, np.ndarray]:
    indices = np.asarray(indices, dtype=np.int64)
    subset = {
        "descriptors": bank.descriptors[indices],
        "position": bank.position[indices],
        "xy": bank.xy[indices],
        "yaw": bank.yaw[indices],
        "indices": bank.indices[indices],
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
    flat_queries = query_tta_descs.reshape(num_queries * num_rot, dim)
    search_k = min(max(int(topk) * 2, int(topk)), int(db_descs.shape[0]))
    flat_scores, flat_indices = search_topk(db_descs, flat_queries, topk=search_k, metric=metric, use_gpu=bool(use_gpu))
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


def _diagnostics_place_payload(rows: list[Dict[str, object]]) -> Dict[str, object]:
    mean_recall_at_1 = float(np.mean([float(row.get("Recall@1", 0.0)) for row in rows])) if rows else 0.0
    mean_recall_at_5 = float(np.mean([float(row.get("Recall@5", 0.0)) for row in rows])) if rows else 0.0
    mean_recall_at_1_percent = float(np.mean([float(row.get("Recall@1%", 0.0)) for row in rows])) if rows else 0.0
    return {
        "Mean Recall@1": mean_recall_at_1,
        "Mean Recall@5": mean_recall_at_5,
        "Mean Recall@1%": mean_recall_at_1_percent,
        "per_sequence": rows,
    }


def _paper_place_payload(rows: list[Dict[str, object]], mean_recall_at_1: float) -> Dict[str, object]:
    return {
        "per_sequence": [{"sequence": str(row["sequence"]), "Recall@1": float(row["Recall@1"])} for row in rows],
        "Mean Recall@1": float(mean_recall_at_1),
    }


def build_shared_frontend_cache(model, cfg: Dict, device: torch.device) -> Dict[str, object]:
    """Extract and cache descriptors needed for evaluation.

    KITTI uses an in-sequence DB/query split. NCLT uses the configured database
    sequence. Pohang uses cross-sequence pairs from the configuration. The cache
    can be reused during training to avoid repeated forward passes over the same
    sequence in every epoch.
    """
    dataset_name = str(cfg.get("dataset", {}).get("name", "kitti")).lower()
    image_size = int(cfg["model"].get("input_size", 201))
    batch_size = int(cfg["evaluation"].get("eval_batch_size", 64))
    query_tta = bool(cfg["evaluation"].get("query_tta", False))
    processed_dataset_kwargs = _build_processed_dataset_kwargs(cfg)
    cache: Dict[str, FeatureBank] = {}
    per_sequence = []

    if dataset_name == "nclt":
        eval_cfg = cfg.get("evaluation", {})
        database_sequence = str(eval_cfg.get("database_sequence", cfg.get("dataset", {}).get("eval_database_sequence", "2012-01-15")))
        for spec in build_nclt_place_specs(
            processed_root=cfg["dataset"]["processed_root"],
            database_sequence=database_sequence,
            query_sequences=_resolve_eval_sequences(cfg),
            positive_radius_m=_get_positive_radius_m(cfg),
        ):
            db_bank = _extract_feature_bank(
                model,
                spec.database_sequence_dir,
                image_size,
                batch_size,
                device,
                processed_dataset_kwargs=processed_dataset_kwargs,
                cache=cache,
                include_tta=False,
            )
            query_bank = _extract_feature_bank(
                model,
                spec.query_sequence_dir,
                image_size,
                batch_size,
                device,
                processed_dataset_kwargs=processed_dataset_kwargs,
                cache=cache,
                include_tta=query_tta,
            )
            per_sequence.append(
                {
                    "sequence": spec.query_sequence_name,
                    "db": _get_subset(db_bank, spec.database_indices),
                    "query": _get_subset(query_bank, spec.query_indices),
                }
            )
    elif dataset_name == "pohang":
        raw_sequence_pairs = cfg.get("evaluation", {}).get("sequence_pairs", None)
        sequence_pairs = None
        if raw_sequence_pairs is not None:
            sequence_pairs = []
            for pair in raw_sequence_pairs:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise RuntimeError(f"Invalid Pohang evaluation.sequence_pairs entry: {pair!r}")
                sequence_pairs.append((str(pair[0]), str(pair[1])))
        for spec in build_pohang_place_specs(
            processed_root=cfg["dataset"]["processed_root"],
            sequence_pairs=sequence_pairs,
            positive_radius_m=_get_positive_radius_m(cfg),
        ):
            db_bank = _extract_feature_bank(
                model,
                spec.database_sequence_dir,
                image_size,
                batch_size,
                device,
                processed_dataset_kwargs=processed_dataset_kwargs,
                cache=cache,
                include_tta=False,
            )
            query_bank = _extract_feature_bank(
                model,
                spec.query_sequence_dir,
                image_size,
                batch_size,
                device,
                processed_dataset_kwargs=processed_dataset_kwargs,
                cache=cache,
                include_tta=query_tta,
            )
            per_sequence.append(
                {
                    "sequence": spec.query_sequence_name,
                    "db": _get_subset(db_bank, spec.database_indices),
                    "query": _get_subset(query_bank, spec.query_indices),
                }
            )
    else:
        for sequence in _resolve_eval_sequences(cfg):
            spec = build_kitti_place_spec(
                processed_root=cfg["dataset"]["processed_root"],
                sequence=str(sequence),
                positive_radius_m=_get_positive_radius_m(cfg),
            )
            bank = _extract_feature_bank(
                model,
                spec.sequence_dir,
                image_size,
                batch_size,
                device,
                processed_dataset_kwargs=processed_dataset_kwargs,
                cache=cache,
                include_tta=query_tta,
            )
            per_sequence.append(
                {
                    "sequence": spec.sequence_name,
                    "db": _get_subset(bank, spec.db_indices),
                    "query": _get_subset(bank, spec.query_indices),
                }
            )

    return {
        "dataset_name": dataset_name,
        "retrieval_metric": str(cfg["evaluation"].get("retrieval_metric", "l2")),
        "retrieval_use_gpu": bool(cfg["evaluation"].get("faiss_gpu", False)),
        "place_positive_radius_m": _get_positive_radius_m(cfg),
        "query_tta": query_tta,
        "query_tta_rotations_deg": [0, 90, 180, 270] if query_tta else [],
        "per_sequence": per_sequence,
    }


def evaluate_place_all(
    model,
    cfg: Dict,
    device: torch.device,
    output_dir: str | Path | None = None,
    shared_frontend: Dict[str, object] | None = None,
) -> Dict[str, object]:
    """Evaluate place recognition on KITTI/NCLT/Pohang.

    Outputs both paper and diagnostics versions: the paper version keeps only
    Recall@1 for tables, while diagnostics also saves Recall@5 and Recall@1%.
    """
    if shared_frontend is None:
        shared_frontend = build_shared_frontend_cache(model, cfg, device)

    eval_cfg = cfg.get("evaluation", {})
    retrieval_metric = str(shared_frontend["retrieval_metric"])
    retrieval_use_gpu = bool(shared_frontend.get("retrieval_use_gpu", False))
    positive_radius_m = float(shared_frontend["place_positive_radius_m"])
    ks = [int(k) for k in eval_cfg.get("place_ks", [1, 5])] or [1, 5]
    use_one_percent = bool(eval_cfg.get("use_one_percent", True))

    rows: list[Dict[str, object]] = []
    for sequence_cache in shared_frontend["per_sequence"]:
        db = sequence_cache["db"]
        query = sequence_cache["query"]
        topk = max(max(ks), max(1, int(round(len(db["indices"]) / 100.0)))) if use_one_percent else max(ks)
        _, predictions = _search_topk_with_optional_tta(
            db["descriptors"],
            query["descriptors"],
            query.get("tta_descriptors", None),
            topk=topk,
            metric=retrieval_metric,
            use_gpu=retrieval_use_gpu,
        )
        recalls = compute_recall_at_k(
            db_xy=db["position"],
            query_xy=query["position"],
            predictions=predictions,
            ks=ks,
            positive_radius_m=positive_radius_m,
        )
        row: Dict[str, object] = {"sequence": sequence_cache["sequence"], **recalls}
        if use_one_percent:
            row["Recall@1%"] = compute_recall_at_one_percent(
                db_xy=db["position"],
                query_xy=query["position"],
                predictions=predictions,
                positive_radius_m=positive_radius_m,
            )
        rows.append(row)

    mean_recall_at_1 = float(np.mean([float(row.get("Recall@1", 0.0)) for row in rows])) if rows else 0.0
    diagnostics_payload = _diagnostics_place_payload(rows)
    paper_payload = _paper_place_payload(rows, mean_recall_at_1)
    summary = {
        "sequences": [str(row["sequence"]) for row in rows],
        "mean_recall_at_1": mean_recall_at_1,
        "mean_recall_at_5": float(diagnostics_payload["Mean Recall@5"]),
        "mean_recall_at_1_percent": float(diagnostics_payload["Mean Recall@1%"]),
        "query_tta": bool(shared_frontend.get("query_tta", False)),
        "query_tta_rotations_deg": list(shared_frontend.get("query_tta_rotations_deg", [])),
        "per_sequence": rows,
        "paper": paper_payload,
        "diagnostics": diagnostics_payload,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        save_json(output_dir / "paper_place.json", paper_payload)
        save_tsv(output_dir / "paper_place.tsv", paper_payload["per_sequence"])
        save_json(output_dir / "diagnostics_place.json", diagnostics_payload)
        save_tsv(output_dir / "diagnostics_place.tsv", diagnostics_payload["per_sequence"])
        save_json(output_dir / "paper_place_all.json", summary)
        save_tsv(
            output_dir / "paper_place_all.tsv",
            diagnostics_payload["per_sequence"]
            + [
                {
                    "sequence": "Mean",
                    "Recall@1": mean_recall_at_1,
                    "Recall@5": diagnostics_payload["Mean Recall@5"],
                    "Recall@1%": diagnostics_payload["Mean Recall@1%"],
                }
            ],
        )
        for row in diagnostics_payload["per_sequence"]:
            seq_dir = ensure_dir(output_dir / str(row["sequence"]))
            seq_payload = {
                "sequence": row["sequence"],
                "Recall@1": float(row["Recall@1"]),
                "Recall@5": float(row["Recall@5"]),
                "Recall@1%": float(row.get("Recall@1%", 0.0)),
                "positive_radius_m": positive_radius_m,
            }
            save_json(seq_dir / "paper_place.json", seq_payload)
            save_tsv(seq_dir / "paper_place.tsv", [seq_payload])
    return summary
