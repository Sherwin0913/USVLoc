#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

GPU_ID="${GPU_ID:-0}"
CONFIG="${CONFIG:-configs/usvloc_default.yaml}"
CHECKPOINT="${CHECKPOINT:-checkpoint/results/final_best_place/usvloc_best_place_recognition.pt}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
DATASET="${DATASET:-kitti}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval_${DATASET}}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python scripts/eval_place.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --set "dataset.name=${DATASET}" \
  --set "dataset.processed_root=${DATA_ROOT}"
