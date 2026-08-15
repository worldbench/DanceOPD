#!/usr/bin/env python3
"""Dependency guard for DanceOPD smoke scripts."""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Requirement:
    label: str
    module: str | None = None
    command: str | None = None
    hint: str = ""
    one_of: tuple[str, ...] = ()


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def has_command(name: str) -> bool:
    return shutil.which(name) is not None


def satisfied(req: Requirement) -> bool:
    if req.one_of:
        return any(has_module(name) or has_command(name) for name in req.one_of)
    ok = True
    if req.module:
        ok = ok and has_module(req.module)
    if req.command:
        ok = ok and has_command(req.command)
    return ok


PYTHON_HINT = "Use the same Python environment that launches the script."

PREPARE_REQUIREMENTS = [
    Requirement(
        "ModelScope downloader",
        one_of=("modelscope",),
        hint='Install with: pip install -e ".[smoke]"  # or ensure the `modelscope` CLI is on PATH.',
    ),
]

DRY_RUN_REQUIREMENTS = [
    Requirement("PyYAML", module="yaml", hint='Install with: pip install -e ".[smoke]"'),
]

TRAIN_COMMON_REQUIREMENTS = [
    Requirement("Accelerate", module="accelerate", hint='Install with: pip install -e ".[smoke]"'),
    Requirement("PyTorch", module="torch", hint="Install a CUDA-enabled PyTorch build for your machine."),
    Requirement("safetensors", module="safetensors", hint='Install with: pip install -e ".[smoke]"'),
    Requirement("tqdm", module="tqdm", hint='Install with: pip install -e ".[smoke]"'),
]

BACKEND_REQUIREMENTS = {
    "toy": [],
    "sd35": [
        Requirement("Diffusers", module="diffusers", hint='Install with: pip install -e ".[sd35,smoke]"'),
        Requirement("Transformers", module="transformers", hint='Install with: pip install -e ".[sd35,smoke]"'),
        Requirement("PEFT", module="peft", hint='Install with: pip install -e ".[sd35,smoke]"'),
    ],
    "zimage": [
        Requirement("Pillow", module="PIL", hint='Install with: pip install -e ".[zimage,smoke]"'),
        Requirement("TorchVision", module="torchvision", hint='Install with: pip install -e ".[zimage,smoke]"'),
        Requirement("PEFT", module="peft", hint='Install with: pip install -e ".[zimage,smoke]"'),
        Requirement(
            "DiffSynth-Studio",
            module="diffsynth",
            hint="Install DiffSynth-Studio following its upstream Z-Image instructions, then rerun this script.",
        ),
    ],
}


def normalize_backend(value: str) -> str:
    backend = value.lower().replace("-", "")
    if backend in {"sd35", "sd3", "stablediffusion35"}:
        return "sd35"
    if backend in {"zimage", "z"}:
        return "zimage"
    if backend in {"toy", "debug", "smoke"}:
        return "toy"
    raise SystemExit(f"Unknown backend: {value!r}. Expected toy, sd35, or zimage.")


def requirements_for(backend: str, level: str, skip_download: bool = False) -> list[Requirement]:
    reqs: list[Requirement] = []
    if not skip_download and level in {"prepare", "dry-run", "train"}:
        reqs.extend(PREPARE_REQUIREMENTS)
    if level in {"dry-run", "train"}:
        reqs.extend(DRY_RUN_REQUIREMENTS)
    if level == "train":
        reqs.extend(TRAIN_COMMON_REQUIREMENTS)
        reqs.extend(BACKEND_REQUIREMENTS[backend])
    return reqs


def report_missing(missing: Iterable[Requirement], backend: str, level: str) -> int:
    missing = list(missing)
    if not missing:
        print(f"[DanceOPD] dependency check passed: backend={backend} level={level}", flush=True)
        return 0

    print(f"[DanceOPD] missing dependencies for backend={backend} level={level}:", file=sys.stderr)
    for req in missing:
        print(f"  - {req.label}: {req.hint}", file=sys.stderr)
    print(f"\n{PYTHON_HINT}", file=sys.stderr)
    if backend == "zimage" and level == "train":
        print("Note: the Z-Image backend imports `diffsynth.pipelines.z_image`; dry-run does not need it, full smoke does.", file=sys.stderr)
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Check dependencies for DanceOPD smoke scripts.")
    parser.add_argument("--backend", default="zimage", help="sd35 or zimage")
    parser.add_argument("--level", choices=["prepare", "dry-run", "train"], default="train")
    parser.add_argument("--skip-download", action="store_true", help="Do not require ModelScope downloader.")
    args = parser.parse_args()

    backend = normalize_backend(args.backend)
    reqs = requirements_for(backend, args.level, skip_download=args.skip_download)
    missing = [req for req in reqs if not satisfied(req)]
    raise SystemExit(report_missing(missing, backend, args.level))


if __name__ == "__main__":
    main()
