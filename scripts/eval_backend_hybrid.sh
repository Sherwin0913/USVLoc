#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

GPU_ID="${GPU_ID:-0}"
CONFIG="${CONFIG:-configs/usvloc_default.yaml}"
USVLOC_CHECKPOINT="${USVLOC_CHECKPOINT:-checkpoint/results/final_best_place/usvloc_best_place_recognition.pt}"
LOCAL_GEOMETRY_CHECKPOINT="${LOCAL_GEOMETRY_CHECKPOINT:-${BEVPLACEPP_CHECKPOINT:-checkpoint/local_geometry_head.pth.tar}}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
DATASETS="${DATASETS:-kitti nclt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/backend_hybrid}"
SKIP_LOOP="${SKIP_LOOP:-0}"
NO_RUNTIME="${NO_RUNTIME:-1}"

read -r -a DATASET_ARGS <<< "${DATASETS}"
EXTRA_ARGS=()
if [[ "${SKIP_LOOP}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-loop)
fi
if [[ "${NO_RUNTIME}" == "1" ]]; then
  EXTRA_ARGS+=(--no-runtime)
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" python scripts/eval_hybrid.py \
  --config "${CONFIG}" \
  --usvloc-ckpt "${USVLOC_CHECKPOINT}" \
  --bevplace-ckpt "${LOCAL_GEOMETRY_CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --processed-root "${DATA_ROOT}" \
  --datasets "${DATASET_ARGS[@]}" \
  --global-rerank-top-k "${GLOBAL_RERANK_TOP_K:-10}" \
  --global-rerank-top-v "${GLOBAL_RERANK_TOP_V:-5}" \
  "${EXTRA_ARGS[@]}" \
  --faiss-gpu
