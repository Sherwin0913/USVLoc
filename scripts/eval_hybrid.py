from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usvloc.backend import SparseRansacBackend, evaluate_backend_bundle, load_hybrid_adapter
from usvloc.io import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    """Parse hybrid backend evaluation arguments.

    Hybrid mode uses USVLoc for global retrieval and BEVPlace++ REM local
    features for geometric verification.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate USVLoc retrieval + BEVPlace++ geometry hybrid backend.",
    )
    parser.add_argument("--usvloc-ckpt", type=Path, required=True)
    parser.add_argument("--bevplace-ckpt", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/usvloc_default.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--dataset", choices=["kitti", "nclt", "pohang", "usvinland"], default=None)
    parser.add_argument("--datasets", nargs="+", default=None, choices=["kitti", "nclt", "pohang", "usvinland"])
    parser.add_argument("--sequence-names", nargs="+", default=None, help="Evaluate only these query sequence names.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")

    parser.add_argument("--image-size", type=int, default=201)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--pair-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-sequences", type=int, default=0, help="Debug only: evaluate only the first N sequences.")
    parser.add_argument("--max-pairs-per-sequence", type=int, default=0, help="Debug only: limit backend pairs per sequence.")
    parser.add_argument("--retrieval-metric", default="l2")
    parser.add_argument("--rerank-top-k", type=int, default=1)
    parser.add_argument("--rerank-top-v", type=int, default=1)
    parser.add_argument("--loop-rerank-top-k", type=int, default=None)
    parser.add_argument("--loop-rerank-top-v", type=int, default=None)
    parser.add_argument("--global-rerank-top-k", type=int, default=None)
    parser.add_argument("--global-rerank-top-v", type=int, default=None)
    parser.add_argument("--rerank-strong-min-inliers", type=int, default=8)
    parser.add_argument("--faiss-gpu", action="store_true")
    parser.add_argument("--disable-query-tta", action="store_true")
    parser.add_argument("--skip-loop", action="store_true")
    parser.add_argument("--skip-global-loc", action="store_true")
    parser.add_argument("--no-runtime", action="store_true")
    parser.add_argument("--runtime-warmup", type=int, default=10)
    parser.add_argument("--runtime-timed-queries", type=int, default=50)

    parser.add_argument("--positive-radius-m", type=float, default=5.0)
    parser.add_argument("--negative-radius-m", type=float, default=15.0)
    parser.add_argument("--success-translation-m", type=float, default=2.0)
    parser.add_argument("--success-rotation-deg", type=float, default=5.0)
    parser.add_argument("--kitti-loader-mode", default="kitti_eval_gray3")

    parser.add_argument("--max-keypoints", type=int, default=0)
    parser.add_argument("--min-keypoints", type=int, default=0)
    parser.add_argument("--fast-threshold", type=int, default=10)
    parser.add_argument("--max-correspondences", type=int, default=0)
    parser.add_argument("--min-correspondences", type=int, default=2)
    parser.add_argument("--min-valid-inliers", type=int, default=0)
    parser.add_argument("--ransac-iterations", type=int, default=1000)
    parser.add_argument("--ransac-threshold-m", type=float, default=0.5)
    parser.add_argument("--score-mode", choices=["inlier_ratio", "num_inliers"], default="inlier_ratio")
    parser.add_argument("--random-seed", type=int, default=1024)
    parser.add_argument("--ransac-threads", type=int, default=4)
    return parser.parse_args()


def _datasets_from_args(args: argparse.Namespace) -> list[str]:
    if args.datasets:
        return [str(item) for item in args.datasets]
    if args.dataset:
        return [str(args.dataset)]
    return ["kitti"]


def main() -> None:
    args = parse_args()
    device_string = f"cuda:{args.gpu}" if args.gpu is not None else str(args.device)
    device = torch.device(device_string)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)

    adapter, metadata = load_hybrid_adapter(
        usvloc_config_path=args.config,
        usvloc_checkpoint_path=args.usvloc_ckpt,
        bevplacepp_checkpoint_path=args.bevplace_ckpt,
        device=device,
        usvloc_overrides=list(args.overrides),
    )
    if bool(args.disable_query_tta):
        adapter.query_uses_tta = False
        metadata["query_tta_rotations_deg"] = []
        metadata["hybrid_note"] = (
            "Global retrieval descriptors are produced by USVLoc without query TTA; "
            "sparse geometric verification uses BEVPlace++ REM local features and the same "
            "BEVPlace2-style RANSAC backend as the BEVPlace++ baseline."
        )
    backend = SparseRansacBackend(
        max_keypoints=int(args.max_keypoints),
        min_keypoints=int(args.min_keypoints),
        fast_threshold=int(args.fast_threshold),
        max_correspondences=int(args.max_correspondences),
        min_correspondences=int(args.min_correspondences),
        min_valid_inliers=int(args.min_valid_inliers),
        ransac_iterations=int(args.ransac_iterations),
        ransac_threshold_m=float(args.ransac_threshold_m),
        score_mode=str(args.score_mode),
        random_seed=int(args.random_seed),
        num_threads=int(args.ransac_threads),
    )

    output_dir = ensure_dir(args.output_dir)
    run_meta = {
        "model_type": "hybrid_usvloc_retrieval_bevplacepp_geometry",
        "usvloc_ckpt": str(args.usvloc_ckpt.resolve()),
        "bevplace_ckpt": str(args.bevplace_ckpt.resolve()),
        "config": str(args.config.resolve()),
        "processed_root": str(args.processed_root.resolve()),
        "datasets": _datasets_from_args(args),
        "sequence_names": None if not args.sequence_names else [str(item) for item in args.sequence_names],
        "device": str(device),
        "faiss_gpu": bool(args.faiss_gpu),
        "query_tta_enabled": int(bool(getattr(adapter, "query_uses_tta", False))),
        "rerank_top_k": int(args.rerank_top_k),
        "rerank_top_v": int(args.rerank_top_v),
        "loop_rerank_top_k": None if args.loop_rerank_top_k is None else int(args.loop_rerank_top_k),
        "loop_rerank_top_v": None if args.loop_rerank_top_v is None else int(args.loop_rerank_top_v),
        "global_rerank_top_k": None if args.global_rerank_top_k is None else int(args.global_rerank_top_k),
        "global_rerank_top_v": None if args.global_rerank_top_v is None else int(args.global_rerank_top_v),
        "rerank_strong_min_inliers": int(args.rerank_strong_min_inliers),
        "run_loop": not bool(args.skip_loop),
        "run_global_loc": not bool(args.skip_global_loc),
        "include_runtime": not bool(args.no_runtime),
        "backend": backend.__dict__,
        "metadata": metadata,
    }
    save_json(output_dir / "run_meta.json", run_meta)

    result = evaluate_backend_bundle(
        adapter=adapter,
        backend=backend,
        datasets=_datasets_from_args(args),
        processed_root=args.processed_root,
        output_dir=output_dir,
        device=device,
        metadata=metadata,
        image_size=int(args.image_size),
        eval_batch_size=int(args.eval_batch_size),
        pair_batch_size=int(args.pair_batch_size),
        num_workers=int(args.num_workers),
        retrieval_metric=str(args.retrieval_metric),
        positive_radius_m=float(args.positive_radius_m),
        negative_radius_m=float(args.negative_radius_m),
        success_translation_m=float(args.success_translation_m),
        success_rotation_deg=float(args.success_rotation_deg),
        kitti_loader_mode=str(args.kitti_loader_mode),
        faiss_gpu=bool(args.faiss_gpu),
        include_runtime=not bool(args.no_runtime),
        runtime_warmup=int(args.runtime_warmup),
        runtime_timed_queries=int(args.runtime_timed_queries),
        sequence_names=None if not args.sequence_names else [str(item) for item in args.sequence_names],
        max_sequences=int(args.max_sequences) if int(args.max_sequences) > 0 else None,
        max_pairs_per_sequence=int(args.max_pairs_per_sequence) if int(args.max_pairs_per_sequence) > 0 else None,
        rerank_top_k=int(args.rerank_top_k),
        rerank_top_v=int(args.rerank_top_v),
        loop_rerank_top_k=None if args.loop_rerank_top_k is None else int(args.loop_rerank_top_k),
        loop_rerank_top_v=None if args.loop_rerank_top_v is None else int(args.loop_rerank_top_v),
        global_rerank_top_k=None if args.global_rerank_top_k is None else int(args.global_rerank_top_k),
        global_rerank_top_v=None if args.global_rerank_top_v is None else int(args.global_rerank_top_v),
        rerank_strong_min_inliers=int(args.rerank_strong_min_inliers),
        run_loop=not bool(args.skip_loop),
        run_global_loc=not bool(args.skip_global_loc),
    )
    print(json.dumps({"output_dir": str(output_dir.resolve()), "summary": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
