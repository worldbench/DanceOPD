#!/usr/bin/env bash
# Shared shell helpers for DanceOPD launch scripts.

resolve_danceopd_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    if "$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
      printf '%s\n' "$PYTHON"
      return 0
    fi
    echo "PYTHON=$PYTHON exists but is older than Python 3.10." >&2
    return 1
  fi

  local candidate
  for candidate in python python3 python3.12 python3.11 python3.10; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
        printf '%s\n' "$candidate"
        return 0
      fi
    fi
  done

  echo "DanceOPD requires Python >= 3.10. Set PYTHON=/path/to/python if it is not on PATH." >&2
  return 1
}
