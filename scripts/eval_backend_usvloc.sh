#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

GPU_ID="${GPU_ID:-0}"
CONFIG="${CONFIG:-configs/usvloc_default.yaml}"
CHECKPOINT="${CHECKPOINT:-checkpoint/results/final_best_place/usvloc_best_place_recognition.pt}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
DATASETS="${DATASETS:-kitti nclt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/backend_usvloc}"

read -r -a DATASET_ARGS <<< "${DATASETS}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python scripts/eval_backend.py \
  --model-type usvloc \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --processed-root "${DATA_ROOT}" \
  --datasets "${DATASET_ARGS[@]}" \
  --faiss-gpu
