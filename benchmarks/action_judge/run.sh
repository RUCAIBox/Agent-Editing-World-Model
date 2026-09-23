#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${EVAL_MODEL:?Set EVAL_MODEL to the served model name}"
export EVAL_BASE_URL="${EVAL_BASE_URL:-http://localhost:8000/v1}"
export EVAL_API_KEY="${EVAL_API_KEY:-EMPTY}"
DATA_ARGS=()
if [[ -n "${DATA_FILE:-}" ]]; then
    DATA_ARGS=(--data-file "${DATA_FILE}")
fi

exec "${PYTHON:-python}" "${ROOT}/benchmarks/action_judge/evaluate.py" \
    --model "${EVAL_MODEL}" \
    --base-url "${EVAL_BASE_URL}" \
    --output-dir "${OUTPUT_DIR:-${ROOT}/results/action_judge}" \
    --concurrency "${CONCURRENCY:-5}" \
    --temperature "${TEMPERATURE:-1.0}" \
    --max-tokens "${MAX_TOKENS:-8192}" \
    --resume "${DATA_ARGS[@]}" "$@"
