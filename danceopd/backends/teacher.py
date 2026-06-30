"""Shared helpers for building frozen DanceOPD teacher fields."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch


Logger = Callable[[str], None]


def load_state_dict_file(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a full/merged model checkpoint state dict.

    Supports `.safetensors` files and common `torch.save` checkpoint wrappers
    such as `{"state_dict": ...}`. LoRA-only adapters should be passed through
    `lora_dir` instead of `base_ckpt`.
    """

    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path))
    else:
        state = torch.load(str(path), map_location="cpu")

    if isinstance(state, dict):
        for key in ("state_dict", "model", "module", "transformer", "dit"):
            value = state.get(key)
            if isinstance(value, dict):
                state = value
                break

    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint did not contain a state dict: {path}")

    # Be permissive about DDP / wrapped checkpoint prefixes.
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = str(key)
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned


def load_full_checkpoint(module: torch.nn.Module, checkpoint: str | Path | None, *, label: str, log: Logger) -> None:
    """Load a non-LoRA, full/merged checkpoint into `module` in-place."""

    if not checkpoint:
        return
    state = load_state_dict_file(checkpoint)
    missing, unexpected = module.load_state_dict(state, strict=False)
    log(
        f"{label} full checkpoint loaded tensors={len(state)} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )


def merge_lora(module: torch.nn.Module, lora_dir: str | Path | None, *, label: str, log: Logger) -> torch.nn.Module:
    """Load and merge a PEFT LoRA adapter into a frozen teacher module."""

    if not lora_dir:
        return module
    from peft import PeftModel

    teacher = PeftModel.from_pretrained(module, str(lora_dir), is_trainable=False)
    teacher = teacher.merge_and_unload()
    log(f"{label} PEFT LoRA merged")
    return teacher


def compose_teacher(
    module: torch.nn.Module,
    teacher_cfg: Any,
    *,
    label: str,
    log: Logger,
    device: torch.device | str,
) -> torch.nn.Module:
    """Apply the public teacher interface and freeze the result.

    Supported cases:
    - `base_ckpt: null`, `lora_dir: null`: the pretrained base model is the teacher.
    - `base_ckpt: /path/full.safetensors`, `lora_dir: null`: a full/merged teacher checkpoint.
    - `base_ckpt: null`, `lora_dir: /path/adapter`: a base model plus PEFT LoRA teacher.
    - both set: load the full base checkpoint first, then merge the PEFT LoRA.
    """

    load_full_checkpoint(module, teacher_cfg.get("base_ckpt"), label=label, log=log)
    module = merge_lora(module, teacher_cfg.get("lora_dir"), label=label, log=log)
    return module.to(device).eval().requires_grad_(False)
