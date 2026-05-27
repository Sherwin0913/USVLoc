#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/backend_default}" \
bash scripts/eval_backend_hybrid.sh
