#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/_env.sh"
PYTHON_BIN=$(resolve_danceopd_python)
BACKEND=${BACKEND:-toy}
NUM_PROCESSES=${NUM_PROCESSES:-1}

case "$BACKEND" in
  toy|debug|smoke)
    BACKEND=toy
    CONFIG=${CONFIG:-configs/smoke/toy_diffsynth_example.yaml}
    SMOKE_SCRIPT=scripts/smoke_toy.sh
    ;;
  sd35|sd3|stable-diffusion-3.5)
    BACKEND=sd35
    CONFIG=${CONFIG:-configs/smoke/sd35_diffsynth_example.yaml}
    SMOKE_SCRIPT=scripts/smoke_sd35.sh
    ;;
  zimage|z-image)
    BACKEND=zimage
    CONFIG=${CONFIG:-configs/smoke/zimage_diffsynth_example.yaml}
    SMOKE_SCRIPT=scripts/smoke_zimage.sh
    ;;
  *)
    echo "Unknown BACKEND=$BACKEND. Expected sd35 or zimage." >&2
    exit 2
    ;;
esac

CHECK_ARGS=(--backend "$BACKEND" --level dry-run)
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

"$PYTHON_BIN" -m danceopd.cli.train --config "$CONFIG" --dry-run

echo "[DanceOPD] bootstrap smoke passed: backend=$BACKEND config=$CONFIG"

SHOULD_RUN_TRAIN=${RUN_TRAIN:-}
if [[ "$BACKEND" == "toy" && -z "$SHOULD_RUN_TRAIN" ]]; then
  SHOULD_RUN_TRAIN=1
fi

if [[ "$SHOULD_RUN_TRAIN" == "1" ]]; then
  echo "[DanceOPD] launching tiny training smoke: backend=$BACKEND"
  DIFFSYNTH_NO_DOWNLOAD=1 NUM_PROCESSES="$NUM_PROCESSES" bash "$SMOKE_SCRIPT" "$@"
else
  echo "[DanceOPD] To run tiny training smoke: RUN_TRAIN=1 BACKEND=$BACKEND bash scripts/bootstrap_smoke.sh"
fi
