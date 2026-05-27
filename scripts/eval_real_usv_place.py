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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usvloc.backend.frontends import load_bevplacepp_adapter, load_usvloc_adapter  # noqa: E402


class BEVFramesDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], image_size: int, divisor: float, force_gray3: bool = False) -> None:
        self.rows = rows
        self.image_size = int(image_size)
        self.divisor = float(divisor)
        self.force_gray3 = bool(force_gray3)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[int(idx)]
        flag = cv2.IMREAD_GRAYSCALE if self.force_gray3 else cv2.IMREAD_COLOR
        image = cv2.imread(str(row["bev_path"]), flag)
        if image is None:
            raise FileNotFoundError(row["bev_path"])
        if self.force_gray3:
            image = np.repeat(image[:, :, None], 3, axis=2)
        elif image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(np.transpose(image.astype(np.float32) / self.divisor, (2, 0, 1))).contiguous()
        return {"image": tensor, "index": int(idx)}


def resolve_bev_path(dataset_root: Path, row: dict[str, Any]) -> Path:
    original = Path(str(row.get("bev_path", "")))
    candidates = [
        dataset_root / "bev" / str(row["sequence"]) / f"{int(row['frame_id']):06d}.png",
        dataset_root / str(row["sequence"]) / f"{int(row['frame_id']):06d}.png",
    ]
    if original.is_absolute():
        candidates.append(original)
        parts = original.parts
        if "bev" in parts:
            idx = parts.index("bev")
            candidates.append(dataset_root.joinpath(*parts[idx:]))
    else:
        candidates.append(dataset_root / original)

    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"Cannot resolve BEV image for sequence={row.get('sequence')} frame_id={row.get('frame_id')}")


def load_real_usv_rows(dataset_root: Path) -> list[dict[str, Any]]:
    csv_path = dataset_root / "frames_with_pose.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing frames_with_pose.csv: {csv_path}")
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sequence", "frame_id", "x_m", "y_m", "z_m", "yaw_rad"}
    missing = required.difference(rows[0].keys() if rows else set())
    if missing:
        raise RuntimeError(f"frames_with_pose.csv is missing required columns: {sorted(missing)}")

    resolved: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        out["bev_path"] = str(resolve_bev_path(dataset_root, row))
        resolved.append(out)
    return resolved


@torch.no_grad()
def extract_descriptors(
    method: str,
    adapter: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
    image_size: int,
    divisor: float,
    force_gray3: bool,
    include_tta: bool,
    num_workers: int,
) -> dict[str, np.ndarray]:
    dataset = BEVFramesDataset(rows, image_size=image_size, divisor=divisor, force_gray3=force_gray3)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
    )
    descs: list[np.ndarray] = []
    tta_descs: list[np.ndarray] = []
    adapter.model.eval()
    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        desc = F.normalize(adapter.forward_global(images), p=2, dim=1)
        descs.append(desc.detach().cpu().numpy().astype(np.float32))
        if include_tta:
            per_rot = [desc]
            for rotation_k in (1, 2, 3):
                rotated_desc = adapter.forward_global(torch.rot90(images, k=rotation_k, dims=(-2, -1)))
                per_rot.append(F.normalize(rotated_desc, p=2, dim=1))
            tta_descs.append(torch.stack(per_rot, dim=1).detach().cpu().numpy().astype(np.float32))
        if (step + 1) % 20 == 0:
            print(f"[{method}] extracted {(step + 1) * batch_size}/{len(dataset)}", flush=True)
    output = {"descriptors": np.concatenate(descs, axis=0)}
    if include_tta:
        output["tta_descriptors"] = np.concatenate(tta_descs, axis=0)
    return output


def search_l2(db: np.ndarray, query: np.ndarray, chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    db = np.ascontiguousarray(db.astype(np.float32, copy=False))
    query = np.ascontiguousarray(query.astype(np.float32, copy=False))
    db_norm = np.sum(db * db, axis=1, dtype=np.float32)[None, :]
    indices = np.empty((query.shape[0],), dtype=np.int64)
    scores = np.empty((query.shape[0],), dtype=np.float32)
    for start in range(0, query.shape[0], chunk):
        end = min(start + chunk, query.shape[0])
        query_chunk = query[start:end]
        dist = np.sum(query_chunk * query_chunk, axis=1, dtype=np.float32)[:, None] + db_norm - 2.0 * (query_chunk @ db.T)
        np.maximum(dist, 0.0, out=dist)
        nearest = np.argmin(dist, axis=1)
        indices[start:end] = nearest
        scores[start:end] = dist[np.arange(end - start), nearest]
    return scores, indices


def search_l2_tta(db: np.ndarray, query_tta: np.ndarray, chunk: int = 128) -> tuple[np.ndarray, np.ndarray]:
    db = np.ascontiguousarray(db.astype(np.float32, copy=False))
    query_tta = np.ascontiguousarray(query_tta.astype(np.float32, copy=False))
    db_norm = np.sum(db * db, axis=1, dtype=np.float32)[None, :]
    indices = np.empty((query_tta.shape[0],), dtype=np.int64)
    scores = np.empty((query_tta.shape[0],), dtype=np.float32)
    for start in range(0, query_tta.shape[0], chunk):
        end = min(start + chunk, query_tta.shape[0])
        flat = query_tta[start:end].reshape(-1, query_tta.shape[-1])
        dist = np.sum(flat * flat, axis=1, dtype=np.float32)[:, None] + db_norm - 2.0 * (flat @ db.T)
        np.maximum(dist, 0.0, out=dist)
        batch = end - start
        dist = dist.reshape(batch, query_tta.shape[1], db.shape[0])
        flat_best = np.argmin(dist.reshape(batch, -1), axis=1)
        rot = flat_best // db.shape[0]
        nearest = flat_best % db.shape[0]
        indices[start:end] = nearest
        scores[start:end] = dist[np.arange(batch), rot, nearest]
    return scores, indices


def evaluate_leave_one_sequence_out(
    rows: list[dict[str, Any]],
    descriptors: np.ndarray,
    tta_descriptors: np.ndarray | None,
    positive_radius_m: float,
) -> list[dict[str, Any]]:
    sequences = sorted({str(row["sequence"]) for row in rows})
    xy = np.asarray([[float(row["x_m"]), float(row["y_m"])] for row in rows], dtype=np.float32)
    seq_arr = np.asarray([str(row["sequence"]) for row in rows])
    results: list[dict[str, Any]] = []
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
            _scores, nn_local = search_l2(descriptors[db_idx], descriptors[query_idx])
        else:
            _scores, nn_local = search_l2_tta(descriptors[db_idx], tta_descriptors[query_idx])
        pred_global = db_idx[nn_local]
        pred_dist = np.linalg.norm(xy[pred_global] - query_xy, axis=1)
        hits = (pred_dist <= radius) & has_positive
        eligible_count = int(has_positive.sum())
        results.append(
            {
                "sequence": sequence,
                "query_frames": int(len(query_idx)),
                "database_frames": int(len(db_idx)),
                "eligible_queries": eligible_count,
                "top1_hits": int(hits.sum()),
                "recall_at_1": float(hits.sum() / max(1, eligible_count)),
                "mean_top1_distance_m": float(np.mean(pred_dist[has_positive])) if eligible_count else float("nan"),
                "median_top1_distance_m": float(np.median(pred_dist[has_positive])) if eligible_count else float("nan"),
            }
        )

    total_eligible = sum(int(row["eligible_queries"]) for row in results)
    total_hits = sum(int(row["top1_hits"]) for row in results)
    results.append(
        {
            "sequence": "Mean",
            "query_frames": sum(int(row["query_frames"]) for row in results),
            "database_frames": "-",
            "eligible_queries": total_eligible,
            "top1_hits": total_hits,
            "recall_at_1": float(total_hits / max(1, total_eligible)),
            "mean_top1_distance_m": float(np.nanmean([float(row["mean_top1_distance_m"]) for row in results])),
            "median_top1_distance_m": float(np.nanmean([float(row["median_top1_distance_m"]) for row in results])),
        }
    )
    return results


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate real-world USV BEV place recognition with leave-one-sequence-out protocol.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/Real-World USV Applications"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval_real_usv_place"))
    parser.add_argument("--usvloc-config", type=Path, default=REPO_ROOT / "configs/usvloc_default.yaml")
    parser.add_argument("--usvloc-ckpt", type=Path, default=REPO_ROOT / "checkpoint/results/final_best_place/usvloc_best_place_recognition.pt")
    parser.add_argument("--bevplace-ckpt", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", default=["USVLoc"], choices=["USVLoc", "BEVPlace++"])
    parser.add_argument("--positive-radius-m", type=float, default=5.0)
    parser.add_argument("--image-size", type=int, default=201)
    parser.add_argument("--batch-size-usvloc", type=int, default=96)
    parser.add_argument("--batch-size-bevplace", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--divisor", type=float, default=255.0)
    parser.add_argument("--force-gray3", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_real_usv_rows(dataset_root)
    print(f"[data] frames={len(rows)} sequences={len(set(row['sequence'] for row in rows))}", flush=True)
    if "pose_delta_ms" in rows[0]:
        deltas = np.asarray([float(row["pose_delta_ms"]) for row in rows], dtype=np.float32)
        print(f"[pose] max_delta_ms={float(deltas.max()):.3f} mean_delta_ms={float(deltas.mean()):.3f}", flush=True)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)

    all_results: dict[str, list[dict[str, Any]]] = {}
    metadata: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "protocol": "leave-one-sequence-out; each sequence is query, the other sequences are database",
        "positive_radius_m": float(args.positive_radius_m),
        "image_size": int(args.image_size),
        "divisor": float(args.divisor),
        "force_gray3": bool(args.force_gray3),
    }

    if "USVLoc" in args.methods:
        print("[model] loading USVLoc", flush=True)
        usvloc, usv_meta = load_usvloc_adapter(args.usvloc_config, args.usvloc_ckpt, device=device)
        print("[extract] USVLoc with 4-rotation query TTA descriptors", flush=True)
        usv_feats = extract_descriptors(
            "USVLoc",
            usvloc,
            rows,
            device=device,
            batch_size=int(args.batch_size_usvloc),
            image_size=int(args.image_size),
            divisor=float(args.divisor),
            force_gray3=bool(args.force_gray3),
            include_tta=True,
            num_workers=int(args.num_workers),
        )
        np.save(output_dir / "usvloc_descriptors.npy", usv_feats["descriptors"])
        np.save(output_dir / "usvloc_tta_descriptors.npy", usv_feats["tta_descriptors"])
        all_results["USVLoc"] = evaluate_leave_one_sequence_out(
            rows,
            usv_feats["descriptors"],
            usv_feats["tta_descriptors"],
            positive_radius_m=float(args.positive_radius_m),
        )
        write_tsv(output_dir / "usvloc_recall_at_1.tsv", all_results["USVLoc"])
        metadata["usvloc"] = usv_meta

    if "BEVPlace++" in args.methods:
        if args.bevplace_ckpt is None:
            raise ValueError("--bevplace-ckpt is required when methods includes BEVPlace++.")
        print("[model] loading BEVPlace++", flush=True)
        bevplace, bev_meta = load_bevplacepp_adapter(args.bevplace_ckpt, device=device)
        print("[extract] BEVPlace++ descriptors", flush=True)
        bev_feats = extract_descriptors(
            "BEVPlace++",
            bevplace,
            rows,
            device=device,
            batch_size=int(args.batch_size_bevplace),
            image_size=int(args.image_size),
            divisor=float(args.divisor),
            force_gray3=bool(args.force_gray3),
            include_tta=False,
            num_workers=int(args.num_workers),
        )
        np.save(output_dir / "bevplacepp_descriptors.npy", bev_feats["descriptors"])
        all_results["BEVPlace++"] = evaluate_leave_one_sequence_out(
            rows,
            bev_feats["descriptors"],
            None,
            positive_radius_m=float(args.positive_radius_m),
        )
        write_tsv(output_dir / "bevplacepp_recall_at_1.tsv", all_results["BEVPlace++"])
        metadata["bevplacepp"] = bev_meta

    combined: list[dict[str, Any]] = []
    for method, rows_for_method in all_results.items():
        for row in rows_for_method:
            out = {"method": method}
            out.update(row)
            combined.append(out)
    write_tsv(output_dir / "place_recognition_recall_at_1_leave_one_sequence_out.tsv", combined)
    (output_dir / "place_recognition_recall_at_1_leave_one_sequence_out.json").write_text(
        json.dumps({**metadata, "results": all_results}, indent=2),
        encoding="utf-8",
    )
    print(f"[done] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
