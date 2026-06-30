#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/_env.sh"
PYTHON_BIN=$(resolve_danceopd_python)
CONFIG=${CONFIG:-configs/smoke/toy_diffsynth_example.yaml}
NUM_PROCESSES=${NUM_PROCESSES:-1}
CHECK_ARGS=(--backend toy --level train)
if [[ -n "${DIFFSYNTH_NO_DOWNLOAD:-}" ]]; then
  CHECK_ARGS+=(--skip-download)
fi

"$PYTHON_BIN" scripts/check_smoke_deps.py "${CHECK_ARGS[@]}"

"$PYTHON_BIN" examples/prepare_diffsynth_example.py \
  --subset "${DIFFSYNTH_SUBSET:-z_image/Z-Image}" \
  --local-dir "${DIFFSYNTH_DATA_DIR:-data/diffsynth_example_dataset}" \
  --output "${DIFFSYNTH_PROMPTS_CSV:-data/diffsynth_example_dataset/danceopd_prompts.csv}" \
  --max-rows "${DIFFSYNTH_MAX_ROWS:-64}" \
  ${DIFFSYNTH_NO_DOWNLOAD:+--no-download}

"$PYTHON_BIN" -m accelerate.commands.launch --num_processes "$NUM_PROCESSES" -m danceopd.cli.train --config "$CONFIG" "$@"
