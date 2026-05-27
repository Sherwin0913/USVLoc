from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usvloc.config import load_config
from usvloc.evaluation import evaluate_usvinland_place
from usvloc.models import USVLoc


def main() -> None:
    """USVInland place recognition evaluation entry point.

    The USVInland protocol reads BEV/INS/Lidar folders directly from
    ``raw_root`` instead of using ``processed_root``.
    """
    parser = argparse.ArgumentParser(description="Evaluate USVLoc on USVInland place recognition using the BEVPlace++ protocol.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--raw-root", default=str(REPO_ROOT / "data/USVInlandRaw"))
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--positive-radius-m", type=float, default=5.0)
    parser.add_argument("--split-ratio", type=float, default=0.5)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--normalization-divisor", type=float, default=255.0)
    parser.add_argument("--faiss-gpu", action="store_true")
    parser.add_argument("--query-tta", action="store_true", help="Use 0/90/180/270 degree query-side TTA for retrieval.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = USVLoc(cfg["model"]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    summary = evaluate_usvinland_place(
        model=model,
        cfg=cfg,
        device=device,
        output_dir=args.output_dir,
        raw_root=args.raw_root,
        sequences=None if args.sequences is None else [str(seq) for seq in args.sequences],
        positive_radius_m=float(args.positive_radius_m),
        split_ratio=float(args.split_ratio),
        eval_batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
        normalization_divisor=float(args.normalization_divisor),
        faiss_gpu=bool(args.faiss_gpu),
        query_tta=bool(args.query_tta),
    )
    summary["config"] = str(Path(args.config).resolve())
    summary["checkpoint"] = str(Path(args.checkpoint).resolve())
    summary["device"] = str(device)
    summary["epoch"] = checkpoint.get("epoch", None) if isinstance(checkpoint, dict) else None
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
