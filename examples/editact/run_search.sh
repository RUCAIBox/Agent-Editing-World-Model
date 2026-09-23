#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${ACTOR_MODEL:?Set ACTOR_MODEL}" "${ACTOR_BASE_URL:?Set ACTOR_BASE_URL}" "${ACTOR_API_KEY:?Set ACTOR_API_KEY}"
: "${WM_MODEL:?Set WM_MODEL}" "${WM_BASE_URL:?Set WM_BASE_URL}"
python examples/editact.py --config configs/editact/search.yaml --task "${TASK:-Use public sources to identify the first release year of Python and name its creator. Read a source before answering.}" --output results/editact/search-example.json "$@"
