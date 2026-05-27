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
from usvloc.evaluation import evaluate_place_all
from usvloc.models import USVLoc


def main() -> None:
    """KITTI/NCLT/Pohang place recognition evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate USVLoc place recognition on KITTI, NCLT, or Pohang.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--query-tta", action="store_true", help="Use 0/90/180/270 degree query-side TTA for retrieval.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    if bool(args.query_tta):
        cfg.setdefault("evaluation", {})["query_tta"] = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = USVLoc(cfg["model"]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    result = evaluate_place_all(model, cfg, device, output_dir=args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
