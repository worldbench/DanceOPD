#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/_env.sh"
PYTHON_BIN=$(resolve_danceopd_python)

CHECK_ARGS=(--level prepare --backend "${BACKEND:-zimage}")
for arg in "$@"; do
  if [[ "$arg" == "--no-download" ]]; then
    CHECK_ARGS+=(--skip-download)
    break
  fi
done

"$PYTHON_BIN" scripts/check_smoke_deps.py "${CHECK_ARGS[@]}"
"$PYTHON_BIN" examples/prepare_diffsynth_example.py "$@"
