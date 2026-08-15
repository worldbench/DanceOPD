"""Checkpoint helpers."""
from __future__ import annotations

import os

import torch


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def step_dir(output_dir: str, step: int) -> str:
    return os.path.join(output_dir, f"step-{step}")


def save_trainer_state(optimizer, output_dir: str, step: int) -> None:
    """Save optimizer/step state without serializing the full base model."""
    torch.save(
        {"step": torch.tensor(int(step)), "optimizer": optimizer.state_dict()},
        os.path.join(output_dir, "trainer_state.pt"),
    )


def load_trainer_state(optimizer, checkpoint_dir: str) -> int:
    path = os.path.join(checkpoint_dir, "trainer_state.pt")
    if not os.path.isfile(path):
        raise ValueError(f"Resume checkpoint has no trainer_state.pt: {checkpoint_dir}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or "optimizer" not in state or "step" not in state:
        raise ValueError(f"Malformed trainer state: {path}")
    optimizer.load_state_dict(state["optimizer"])
    step = state["step"]
    return int(step.item() if isinstance(step, torch.Tensor) else step)
