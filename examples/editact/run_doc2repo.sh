#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${ACTOR_MODEL:?Set ACTOR_MODEL}" "${ACTOR_BASE_URL:?Set ACTOR_BASE_URL}" "${ACTOR_API_KEY:?Set ACTOR_API_KEY}"
: "${WM_MODEL:?Set WM_MODEL}" "${WM_BASE_URL:?Set WM_BASE_URL}"
python examples/editact.py --config configs/editact/doc2repo.yaml --task "${TASK:-Implement a small Python package tiny_stats in /workspace. Its public API mean(values) returns the arithmetic mean and raises ValueError on empty input. Include a pyproject.toml and unittest tests. Execute the tests before finishing.}" --output results/editact/doc2repo-example.json "$@"
