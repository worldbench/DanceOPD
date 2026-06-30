#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/_env.sh"
PYTHON_BIN=$(resolve_danceopd_python)

"$PYTHON_BIN" examples/prepare_omniedit.py "$@"
