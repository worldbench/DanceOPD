#!/usr/bin/env bash
set -euo pipefail
BACKEND="${1:-sd35}"
METHOD="${2:-danceopd}"
case "${BACKEND}" in sd35|zimage) ;; *) echo "backend must be sd35 or zimage" >&2; exit 2;; esac
case "${METHOD}" in danceopd|diffusionopd|flowopd) ;; *) echo "method must be danceopd, diffusionopd, or flowopd" >&2; exit 2;; esac
CONFIG="configs/public/${BACKEND}.yaml"
if [[ "${BACKEND}" == "sd35" && "${METHOD}" == "flowopd" ]]; then
  CONFIG="configs/public/sd35_flowopd.yaml"
fi
PYTHON_BIN="${PYTHON:-python3}"
exec "${PYTHON_BIN}" -m danceopd.cli.train --config "${CONFIG}" --set "training.method=${METHOD}" "${@:3}"
