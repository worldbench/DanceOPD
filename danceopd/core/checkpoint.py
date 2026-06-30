"""Checkpoint helpers."""
from __future__ import annotations

import os


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def step_dir(output_dir: str, step: int) -> str:
    return os.path.join(output_dir, f"step-{step}")
