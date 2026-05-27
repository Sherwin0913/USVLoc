#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

GPU_ID="${GPU_ID:-0}"
CONFIG="${CONFIG:-configs/usvloc_default.yaml}"
CHECKPOINT="${CHECKPOINT:-checkpoint/results/final_best_place/usvloc_best_place_recognition.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train_resume}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
LOAD_OPTIMIZER="${LOAD_OPTIMIZER:-1}"
EXTRA_SET_ARGS="${EXTRA_SET_ARGS:-}"

mkdir -p "$(dirname "${OUTPUT_DIR}")"

EXTRA_ARGS=()
if [[ "${LOAD_OPTIMIZER}" == "0" ]]; then
  EXTRA_ARGS+=(--no-resume-optimizer)
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" python scripts/train.py \
  --config "${CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --resume-checkpoint "${CHECKPOINT}" \
  --set "dataset.processed_root=${DATA_ROOT}" \
  --set "evaluation.cross_dataset_eval.processed_root=${DATA_ROOT}" \
  "${EXTRA_ARGS[@]}" \
  ${EXTRA_SET_ARGS}
