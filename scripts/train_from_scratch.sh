#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

GPU_ID="${GPU_ID:-0}"
CONFIG="${CONFIG:-configs/usvloc_default.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train_from_scratch}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$(dirname "${OUTPUT_DIR}")"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python scripts/train.py \
  --config "${CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --set "dataset.processed_root=${DATA_ROOT}" \
  --set "evaluation.cross_dataset_eval.processed_root=${DATA_ROOT}" \
  ${EXTRA_ARGS}
