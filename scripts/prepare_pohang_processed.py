from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _load_baseline(sequence_root: Path) -> pd.DataFrame:
    with zipfile.ZipFile(sequence_root / "navigation.zip") as zf:
        with zf.open("navigation/baseline.txt") as handle:
            baseline = pd.read_csv(handle, sep="\t", header=None)
    baseline.columns = [
        "timestamp_s",
        "aux_1",
        "aux_2",
        "heading_x",
        "heading_y",
        "north_m",
        "east_m",
        "aux_heading",
    ]
    return baseline


def _align_sequence(
    dataset_root: Path,
    sequence: str,
    bev_subdir: str,
    sample_interval: int,
    max_nav_time_diff_s: float,
) -> pd.DataFrame:
    sequence_root = dataset_root / sequence
    bev_paths = sorted((sequence_root / bev_subdir).glob("*.png"))
    if not bev_paths:
        raise FileNotFoundError(f"No BEV PNGs found for {sequence}: {sequence_root / bev_subdir}")

    baseline = _load_baseline(sequence_root)
    baseline_ts = baseline["timestamp_s"].to_numpy(dtype=np.float64)
    east = baseline["east_m"].to_numpy(dtype=np.float64)
    north = baseline["north_m"].to_numpy(dtype=np.float64)
    yaw = np.arctan2(
        baseline["heading_y"].to_numpy(dtype=np.float64),
        baseline["heading_x"].to_numpy(dtype=np.float64),
    )

    bev_ts = np.asarray([int(path.stem) / 1e9 for path in bev_paths], dtype=np.float64)
    right = np.searchsorted(baseline_ts, bev_ts, side="left")
    right = np.clip(right, 1, len(baseline_ts) - 1)
    left = right - 1
    use_right = np.abs(baseline_ts[right] - bev_ts) < np.abs(baseline_ts[left] - bev_ts)
    match_idx = np.where(use_right, right, left)
    match_dt = np.abs(baseline_ts[match_idx] - bev_ts)

    frames = pd.DataFrame(
        {
            "frame_id": [path.stem for path in bev_paths],
            "timestamp": bev_ts,
            "bev_path": [str(path.resolve()) for path in bev_paths],
            "nav_match_dt_s": match_dt,
            "east_m": east[match_idx],
            "north_m": north[match_idx],
            "yaw_rad": yaw[match_idx],
        }
    )
    frames = frames[frames["nav_match_dt_s"] <= float(max_nav_time_diff_s)].reset_index(drop=True)
    frames = frames.iloc[:: int(sample_interval)].reset_index(drop=True)
    if frames.empty:
        raise RuntimeError(f"No frames survived alignment for {sequence}")
    return frames


def _write_sequence(output_dir: Path, sequence: str, frames: pd.DataFrame, origin_east: float, origin_north: float) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = frames.copy()
    frames["x_m"] = frames["east_m"] - float(origin_east)
    frames["y_m"] = frames["north_m"] - float(origin_north)
    frames["z_m"] = 0.0
    frames["split_tag"] = "eval"

    keep_cols = [
        "frame_id",
        "timestamp",
        "bev_path",
        "x_m",
        "y_m",
        "z_m",
        "yaw_rad",
        "split_tag",
        "nav_match_dt_s",
        "east_m",
        "north_m",
    ]
    frames[keep_cols].to_csv(output_dir / "frames.csv", index=False)

    meta = {
        "dataset_name": "pohang",
        "sequence_name": sequence,
        "pose_adapter": "xyyaw",
        "x_range_m": [-80.0, 80.0],
        "y_range_m": [-80.0, 80.0],
        "meters_per_pixel": 160.0 / 256.0,
        "source_image_height": 256,
        "source_image_width": 256,
        "model_input_size": 201,
        "input_channels": 3,
        "normalization_divisor": 255.0,
        "notes": (
            "Pohang Canal fused BEV; yaw from baseline heading_x/heading_y; "
            "x/y use one shared east/north origin across all Pohang sequences."
        ),
    }
    with (output_dir / "meta.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(meta, handle, sort_keys=False, allow_unicode=True)

    return {
        "sequence_name": sequence,
        "output_dir": str(output_dir.resolve()),
        "num_frames": int(len(frames)),
        "max_nav_match_dt_s": float(frames["nav_match_dt_s"].max()),
        "bev_dir": str(Path(frames["bev_path"].iloc[0]).parent.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare USVLoc processed Pohang sequences from fused BEV PNGs.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/PohangRaw"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/Pohang"))
    parser.add_argument("--sequences", nargs="+", default=[f"pohang{i:02d}" for i in range(6)])
    parser.add_argument("--bev-subdir", default="bev_fused_256")
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--max-nav-time-diff-s", type=float, default=0.5)
    args = parser.parse_args()

    sequence_frames = {
        sequence: _align_sequence(
            dataset_root=args.dataset_root,
            sequence=str(sequence),
            bev_subdir=str(args.bev_subdir),
            sample_interval=int(args.sample_interval),
            max_nav_time_diff_s=float(args.max_nav_time_diff_s),
        )
        for sequence in args.sequences
    }
    origin_east = min(float(frames["east_m"].min()) for frames in sequence_frames.values())
    origin_north = min(float(frames["north_m"].min()) for frames in sequence_frames.values())

    summaries = []
    for sequence, frames in sequence_frames.items():
        summaries.append(_write_sequence(args.processed_root / sequence, sequence, frames, origin_east, origin_north))

    summary = {
        "dataset_root": str(args.dataset_root.resolve()),
        "processed_root": str(args.processed_root.resolve()),
        "bev_subdir": str(args.bev_subdir),
        "sample_interval": int(args.sample_interval),
        "max_nav_time_diff_s": float(args.max_nav_time_diff_s),
        "shared_origin": {"east_m": origin_east, "north_m": origin_north},
        "sequences": summaries,
    }
    args.processed_root.mkdir(parents=True, exist_ok=True)
    (args.processed_root / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
