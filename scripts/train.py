from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usvloc.config import load_config
from usvloc.training import train_usvloc


def main() -> None:
    """Training command-line entry point.

    This script only parses YAML, command-line overrides, and checkpoint
    resume arguments. The training logic lives in
    ``usvloc/training/train.py::train_usvloc``.
    """
    parser = argparse.ArgumentParser(description="Train the final USVLoc architecture.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--output-dir", default=None, help="Optional explicit run directory.")
    parser.add_argument("--resume-checkpoint", default=None, help="Resume model/optimizer from a checkpoint.")
    parser.add_argument(
        "--no-resume-optimizer",
        action="store_true",
        help="Only load model weights from --resume-checkpoint and reinitialize optimizer.",
    )
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    run_dir = train_usvloc(
        cfg,
        output_dir=args.output_dir,
        resume_checkpoint=args.resume_checkpoint,
        load_optimizer=not bool(args.no_resume_optimizer),
    )
    print(json.dumps({"run_dir": str(run_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
