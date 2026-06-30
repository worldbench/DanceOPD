#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/_env.sh"
PYTHON_BIN=$(resolve_danceopd_python)
CONFIG=${CONFIG:-configs/zimage_danceopd.yaml}
NUM_PROCESSES=${NUM_PROCESSES:-1}

"$PYTHON_BIN" -m accelerate.commands.launch --num_processes "$NUM_PROCESSES" -m danceopd.cli.train --config "$CONFIG" "$@"
