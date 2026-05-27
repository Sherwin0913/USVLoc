#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usvloc.backend import SparseRansacBackend, load_bevplacepp_adapter, load_hybrid_adapter  # noqa: E402
from usvloc.backend.evaluator import _relative_pose_error  # noqa: E402

from eval_real_usv_place import load_real_usv_rows, search_l2, search_l2_tta  # noqa: E402


def save_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def load_image_tensor(path: str | Path, image_size: int, divisor: float, force_gray3: bool = False) -> torch.Tensor:
    flag = cv2.IMREAD_GRAYSCALE if force_gray3 else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise FileNotFoundError(path)
    if force_gray3:
        image = np.repeat(image[:, :, None], 3, axis=2)
    elif image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[0] != image_size or image.shape[1] != image_size:
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / float(divisor)
    return torch.from_numpy(np.transpose(image, (2, 0, 1))).contiguous()


def build_top1_details(
    rows: list[dict[str, Any]],
    descriptors: np.ndarray,
    tta_descriptors: np.ndarray | None,
    positive_radius_m: float,
) -> dict[str, list[dict[str, Any]]]:
    sequences = sorted({str(row["sequence"]) for row in rows})
    seq_arr = np.asarray([str(row["sequence"]) for row in rows])
    xy = np.asarray([[float(row["x_m"]), float(row["y_m"])] for row in rows], dtype=np.float32)
    output: dict[str, list[dict[str, Any]]] = {}
    radius = float(positive_radius_m)
    for sequence in sequences:
        query_idx = np.where(seq_arr == sequence)[0]
        db_idx = np.where(seq_arr != sequence)[0]
        query_xy = xy[query_idx]
        db_xy = xy[db_idx]
        has_positive = np.zeros(len(query_idx), dtype=bool)
        for start in range(0, len(query_idx), 512):
            end = min(start + 512, len(query_idx))
            dist = np.linalg.norm(query_xy[start:end, None, :] - db_xy[None, :, :], axis=2)
            has_positive[start:end] = np.any(dist <= radius, axis=1)
        if tta_descriptors is None:
            scores, nn_local = search_l2(descriptors[db_idx], descriptors[query_idx])
        else:
            scores, nn_local = search_l2_tta(descriptors[db_idx], tta_descriptors[query_idx])
        pred_global = db_idx[nn_local]
        pred_dist = np.linalg.norm(xy[pred_global] - query_xy, axis=1)
        details: list[dict[str, Any]] = []
        for local_i, global_query in enumerate(query_idx):
            if not bool(has_positive[local_i]):
                continue
            global_db = int(pred_global[local_i])
            details.append(
                {
                    "sequence": sequence,
                    "query_global_index": int(global_query),
                    "db_global_index": global_db,
                    "descriptor_score": float(scores[local_i]),
                    "retrieval_distance_m": float(pred_dist[local_i]),
                    "retrieval_hit": int(pred_dist[local_i] <= radius),
                }
            )
        output[sequence] = details
    return output


@torch.no_grad()
def evaluate_pairs_for_sequence(
    method: str,
    adapter: Any,
    backend: SparseRansacBackend,
    rows: list[dict[str, Any]],
    details: list[dict[str, Any]],
    device: torch.device,
    image_size: int,
    divisor: float,
    pair_batch_size: int,
    meters_per_pixel: float,
    success_translation_m: float,
    success_rotation_deg: float,
    force_gray3: bool,
    seed_base: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pair_rows: list[dict[str, Any]] = []
    for start in range(0, len(details), int(pair_batch_size)):
        end = min(start + int(pair_batch_size), len(details))
        batch = details[start:end]
        query_images = torch.stack(
            [
                load_image_tensor(rows[int(item["query_global_index"])]["bev_path"], image_size, divisor, force_gray3=force_gray3)
                for item in batch
            ],
            dim=0,
        )
        candidate_images = torch.stack(
            [
                load_image_tensor(rows[int(item["db_global_index"])]["bev_path"], image_size, divisor, force_gray3=force_gray3)
                for item in batch
            ],
            dim=0,
        )
        _, _, query_local, candidate_local = adapter.forward_pair_features(
            query_images.to(device, non_blocking=True),
            candidate_images.to(device, non_blocking=True),
        )
        query_images_cpu = query_images.cpu()
        candidate_images_cpu = candidate_images.cpu()
        query_local_cpu = query_local.detach().cpu()
        candidate_local_cpu = candidate_local.detach().cpu()
        for batch_i, item in enumerate(batch):
            query_i = int(item["query_global_index"])
            db_i = int(item["db_global_index"])
            np.random.seed(int(seed_base) + query_i * 100003 + db_i)
            result = backend.solve_one(
                query_image=query_images_cpu[batch_i],
                candidate_image=candidate_images_cpu[batch_i],
                query_local=query_local_cpu[batch_i],
                candidate_local=candidate_local_cpu[batch_i],
                meters_per_pixel=float(meters_per_pixel),
            )
            query_row = rows[query_i]
            db_row = rows[db_i]
            e_t = None
            e_r = None
            success = 0
            if bool(result.pose_valid):
                e_t, e_r = _relative_pose_error(
                    result,
                    query_xy=np.asarray([float(query_row["x_m"]), float(query_row["y_m"])], dtype=np.float32),
                    query_yaw=float(query_row["yaw_rad"]),
                    candidate_xy=np.asarray([float(db_row["x_m"]), float(db_row["y_m"])], dtype=np.float32),
                    candidate_yaw=float(db_row["yaw_rad"]),
                )
                success = int(float(e_t) < float(success_translation_m) and float(e_r) < float(success_rotation_deg))
            pair_rows.append(
                {
                    "method": method,
                    "sequence": item["sequence"],
                    "query_global_index": query_i,
                    "query_frame_id": int(rows[query_i]["frame_id"]),
                    "db_global_index": db_i,
                    "db_sequence": rows[db_i]["sequence"],
                    "db_frame_id": int(rows[db_i]["frame_id"]),
                    "retrieval_hit": int(item["retrieval_hit"]),
                    "retrieval_distance_m": float(item["retrieval_distance_m"]),
                    "pose_valid": int(result.pose_valid),
                    "success": int(success),
                    "e_t_m": "" if e_t is None else float(e_t),
                    "e_r_deg": "" if e_r is None else float(e_r),
                    "num_inliers": int(result.num_inliers),
                    "num_matches": int(result.num_matches),
                    "backend_score": float(result.score),
                }
            )
        if end == len(details) or end % 200 == 0:
            print(f"[{method}] {details[0]['sequence']} pose pairs {end}/{len(details)}", flush=True)

    success_rows = [row for row in pair_rows if int(row["success"]) == 1]
    valid_rows = [row for row in pair_rows if int(row["pose_valid"]) == 1]
    summary = {
        "method": method,
        "sequence": details[0]["sequence"] if details else "",
        "eligible_queries": int(len(details)),
        "retrieval_hits": int(sum(int(item["retrieval_hit"]) for item in details)),
        "Recall@1": float(sum(int(item["retrieval_hit"]) for item in details)) / float(max(1, len(details))),
        "successful_queries": int(len(success_rows)),
        "SR": float(len(success_rows)) / float(max(1, len(details))),
        "e_t_m": float(np.mean([float(row["e_t_m"]) for row in success_rows])) if success_rows else 0.0,
        "e_r_deg": float(np.mean([float(row["e_r_deg"]) for row in success_rows])) if success_rows else 0.0,
        "pose_valid_queries": int(len(valid_rows)),
        "pose_valid_rate": float(len(valid_rows)) / float(max(1, len(details))),
        "valid_e_t_m": float(np.mean([float(row["e_t_m"]) for row in valid_rows])) if valid_rows else 0.0,
        "valid_e_r_deg": float(np.mean([float(row["e_r_deg"]) for row in valid_rows])) if valid_rows else 0.0,
    }
    return summary, pair_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate real-world USV global localization from Top-1 retrieval pairs.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/Real-World USV Applications"))
    parser.add_argument("--place-dir", type=Path, default=Path("outputs/eval_real_usv_place"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval_real_usv_global_localization"))
    parser.add_argument("--usvloc-config", type=Path, default=REPO_ROOT / "configs/usvloc_default.yaml")
    parser.add_argument("--usvloc-ckpt", type=Path, default=REPO_ROOT / "checkpoint/results/final_best_place/usvloc_best_place_recognition.pt")
    parser.add_argument("--local-geometry-ckpt", type=Path, default=None)
    parser.add_argument("--bevplace-ckpt", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", default=["USVLoc"], choices=["USVLoc", "BEVPlace++"])
    parser.add_argument("--positive-radius-m", type=float, default=5.0)
    parser.add_argument("--success-translation-m", type=float, default=2.0)
    parser.add_argument("--success-rotation-deg", type=float, default=5.0)
    parser.add_argument("--image-size", type=int, default=201)
    parser.add_argument("--meters-per-pixel", type=float, default=0.4)
    parser.add_argument("--divisor", type=float, default=255.0)
    parser.add_argument("--pair-batch-size", type=int, default=4)
    parser.add_argument("--ransac-iterations", type=int, default=1000)
    parser.add_argument("--ransac-threshold-m", type=float, default=0.5)
    parser.add_argument("--fast-threshold", type=int, default=10)
    parser.add_argument("--force-gray3", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.local_geometry_ckpt is None and args.bevplace_ckpt is None:
        raise SystemExit("Please provide --local-geometry-ckpt.")
    geometry_ckpt = args.local_geometry_ckpt if args.local_geometry_ckpt is not None else args.bevplace_ckpt
    dataset_root = args.dataset_root.resolve()
    place_dir = args.place_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_real_usv_rows(dataset_root)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)

    descs: dict[str, np.ndarray] = {}
    if "USVLoc" in args.methods:
        descs["USVLoc"] = np.load(place_dir / "usvloc_descriptors.npy")
        descs["USVLoc_TTA"] = np.load(place_dir / "usvloc_tta_descriptors.npy")
    if "BEVPlace++" in args.methods:
        descs["BEVPlace++"] = np.load(place_dir / "bevplacepp_descriptors.npy")
    print(f"[data] rows={len(rows)} device={device}", flush=True)

    top1: dict[str, dict[str, list[dict[str, Any]]]] = {}
    if "USVLoc" in args.methods:
        top1["USVLoc"] = build_top1_details(rows, descs["USVLoc"], descs["USVLoc_TTA"], args.positive_radius_m)
    if "BEVPlace++" in args.methods:
        top1["BEVPlace++"] = build_top1_details(rows, descs["BEVPlace++"], None, args.positive_radius_m)

    backend = SparseRansacBackend(
        max_keypoints=0,
        min_keypoints=0,
        fast_threshold=int(args.fast_threshold),
        max_correspondences=0,
        min_correspondences=2,
        min_valid_inliers=0,
        ransac_iterations=int(args.ransac_iterations),
        ransac_threshold_m=float(args.ransac_threshold_m),
        score_mode="inlier_ratio",
        random_seed=1024,
        num_threads=1,
    )

    adapters: dict[str, Any] = {}
    if "USVLoc" in args.methods:
        adapters["USVLoc"], _ = load_hybrid_adapter(args.usvloc_config, args.usvloc_ckpt, geometry_ckpt, device=device)
    if "BEVPlace++" in args.methods:
        adapters["BEVPlace++"], _ = load_bevplacepp_adapter(geometry_ckpt, device=device)

    all_summary: list[dict[str, Any]] = []
    for method in args.methods:
        method_dir = output_dir / method.lower().replace("+", "p").replace(" ", "_")
        method_dir.mkdir(parents=True, exist_ok=True)
        for sequence, details in top1[method].items():
            summary_path = method_dir / f"{sequence}_summary.json"
            pairs_path = method_dir / f"{sequence}_pairs.tsv"
            if args.skip_existing and summary_path.exists() and pairs_path.exists():
                all_summary.append(json.loads(summary_path.read_text(encoding="utf-8")))
                continue
            print(f"[eval] method={method} seq={sequence} eligible={len(details)}", flush=True)
            summary, pair_rows = evaluate_pairs_for_sequence(
                method=method,
                adapter=adapters[method],
                backend=backend,
                rows=rows,
                details=details,
                device=device,
                image_size=int(args.image_size),
                divisor=float(args.divisor),
                pair_batch_size=int(args.pair_batch_size),
                meters_per_pixel=float(args.meters_per_pixel),
                success_translation_m=float(args.success_translation_m),
                success_rotation_deg=float(args.success_rotation_deg),
                force_gray3=bool(args.force_gray3),
                seed_base=1024 if method == "USVLoc" else 4096,
            )
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            save_tsv(pairs_path, pair_rows)
            all_summary.append(summary)
            print(
                f"[done] {method} {sequence} SR={summary['SR']:.4f} e_t={summary['e_t_m']:.3f} e_r={summary['e_r_deg']:.3f}",
                flush=True,
            )

    final_rows = list(all_summary)
    for method in args.methods:
        rows_for_method = [row for row in all_summary if row["method"] == method]
        denom = sum(int(row["eligible_queries"]) for row in rows_for_method)
        successes = sum(int(row["successful_queries"]) for row in rows_for_method)
        success_weighted_et: list[float] = []
        success_weighted_er: list[float] = []
        for row in rows_for_method:
            if int(row["successful_queries"]) > 0:
                success_weighted_et.extend([float(row["e_t_m"])] * int(row["successful_queries"]))
                success_weighted_er.extend([float(row["e_r_deg"])] * int(row["successful_queries"]))
        final_rows.append(
            {
                "method": method,
                "sequence": "Mean weighted",
                "eligible_queries": int(denom),
                "retrieval_hits": sum(int(row["retrieval_hits"]) for row in rows_for_method),
                "Recall@1": sum(int(row["retrieval_hits"]) for row in rows_for_method) / float(max(1, denom)),
                "successful_queries": int(successes),
                "SR": int(successes) / float(max(1, denom)),
                "e_t_m": float(np.mean(success_weighted_et)) if success_weighted_et else 0.0,
                "e_r_deg": float(np.mean(success_weighted_er)) if success_weighted_er else 0.0,
                "pose_valid_queries": sum(int(row["pose_valid_queries"]) for row in rows_for_method),
                "pose_valid_rate": sum(int(row["pose_valid_queries"]) for row in rows_for_method) / float(max(1, denom)),
                "valid_e_t_m": 0.0,
                "valid_e_r_deg": 0.0,
            }
        )

    save_tsv(output_dir / "global_localization_top1_summary.tsv", final_rows)
    (output_dir / "global_localization_top1_summary.json").write_text(json.dumps(final_rows, indent=2), encoding="utf-8")
    (output_dir / "protocol.json").write_text(
        json.dumps(
            {
                "protocol": "leave-one-sequence-out; Top-1 retrieved database frame is geometrically verified",
                "dataset_root": str(dataset_root),
                "place_dir": str(place_dir),
                "positive_radius_m": float(args.positive_radius_m),
                "success_translation_m": float(args.success_translation_m),
                "success_rotation_deg": float(args.success_rotation_deg),
                "usvloc_backend": "USVLoc descriptor retrieval with 4-rotation query TTA; BEVPlace++ REM local features and RANSAC for pose",
                "bevplacepp_backend": "BEVPlace++ descriptor retrieval; BEVPlace++ REM local features and RANSAC for pose",
                "errors": "e_t/e_r are averaged over successful localizations.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] wrote {output_dir / 'global_localization_top1_summary.tsv'}", flush=True)


if __name__ == "__main__":
    main()
