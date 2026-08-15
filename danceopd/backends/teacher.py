"""Shared helpers for building frozen DanceOPD teacher fields."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

Logger = Callable[[str], None]


def resolve_checkpoint(value: str | Path) -> str:
    """Resolve local paths or ``hf://repo_id/path`` without custom scripts."""
    text = str(value)
    if not text.startswith("hf://"):
        return text
    spec = text[len("hf://") :]
    parts = spec.split("/")
    if len(parts) < 3:
        raise ValueError(f"HF checkpoint must be hf://owner/repo/file: {text}")
    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename)


def load_state_dict_file(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a full/merged model checkpoint state dict.

    Supports `.safetensors` files and common `torch.save` checkpoint wrappers
    such as `{"state_dict": ...}`. LoRA-only adapters should be passed through
    `lora_dir` instead of `base_ckpt`.
    """

    path = Path(resolve_checkpoint(path))
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path))
    else:
        try:
            state = torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError as exc:
            raise RuntimeError(
                "Safe checkpoint loading requires a PyTorch version that supports "
                "torch.load(..., weights_only=True). Upgrade PyTorch or convert the file to safetensors."
            ) from exc

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
            new_key = new_key.removeprefix(prefix)
        cleaned[new_key] = value
    return cleaned


def load_compatible_state_dict(
    module: torch.nn.Module,
    state: dict[str, torch.Tensor],
    *,
    label: str,
    log: Logger,
    min_model_match: float = 0.5,
) -> int:
    """Load only shape-compatible tensors and reject ineffective checkpoints."""

    expected = module.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in expected and isinstance(value, torch.Tensor) and expected[key].shape == value.shape
    }
    ratio = len(compatible) / max(1, len(expected))
    if not compatible or ratio < min_model_match:
        raise RuntimeError(
            f"{label} is incompatible with this model: matched {len(compatible)}/{len(expected)} "
            f"model tensors ({ratio:.1%}); required at least {min_model_match:.1%}."
        )
    missing, unexpected = module.load_state_dict(compatible, strict=False)
    log(
        f"{label} loaded compatible={len(compatible)}/{len(expected)} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    return len(compatible)


def load_full_checkpoint(module: torch.nn.Module, checkpoint: str | Path | None, *, label: str, log: Logger) -> None:
    """Load a non-LoRA, full/merged checkpoint into `module` in-place."""

    if not checkpoint:
        return
    state = load_state_dict_file(checkpoint)
    load_compatible_state_dict(module, state, label=f"{label} full checkpoint", log=log)


def merge_lora(module: torch.nn.Module, lora_dir: str | Path | None, *, label: str, log: Logger) -> torch.nn.Module:
    """Load and merge a PEFT LoRA adapter into a frozen teacher module."""

    if not lora_dir:
        return module
    resolved = resolve_checkpoint(lora_dir)
    if Path(resolved).is_file():
        raise ValueError(
            f"{label} received a single-file LoRA ({resolved}). This generic backend cannot infer "
            "its rank and target modules safely. Provide a PEFT adapter directory/model ID, or use "
            "the Z-Image backend's explicit raw-LoRA fields."
        )
    from peft import PeftModel

    teacher = PeftModel.from_pretrained(module, resolved, is_trainable=False)
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
