#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${ACTOR_MODEL:?Set ACTOR_MODEL}" "${ACTOR_BASE_URL:?Set ACTOR_BASE_URL}" "${ACTOR_API_KEY:?Set ACTOR_API_KEY}"
: "${WM_MODEL:?Set WM_MODEL}" "${WM_BASE_URL:?Set WM_BASE_URL}"
python examples/editact.py --config configs/editact/terminal.yaml --task "${TASK:-In /workspace, write numbers.txt containing 1, 2, 3 on separate lines, and a Python script that reads it and writes their sum to total.txt. Run and verify the script.}" --output results/editact/terminal-example.json "$@"
