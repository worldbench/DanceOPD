#!/usr/bin/env python3
"""Prepare a prompt CSV from the official DiffSynth example dataset.

The script downloads (or reuses) `DiffSynth-Studio/diffsynth_example_dataset`
through ModelScope and extracts prompts from the selected subset metadata.
It intentionally stores caches under the project data directory instead of a
user home cache, so it is friendly to shared clusters and containers.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROMPT_KEYS = (
    "prompt",
    "caption",
    "text",
    "instruction",
    "edit_prompt",
    "target_prompt",
    "description",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download DiffSynth example metadata and extract prompts.")
    parser.add_argument("--dataset-id", default="DiffSynth-Studio/diffsynth_example_dataset")
    parser.add_argument("--subset", default="z_image/Z-Image", help="Subset folder inside the official example dataset.")
    parser.add_argument("--local-dir", default="data/diffsynth_example_dataset")
    parser.add_argument("--include", default=None, help="Override download include pattern. Default: '<subset>/*'.")
    parser.add_argument("--output", default="data/diffsynth_example_dataset/danceopd_prompts.csv")
    parser.add_argument("--max-rows", type=int, default=64)
    parser.add_argument("--no-download", action="store_true", help="Only parse an already downloaded dataset.")
    return parser.parse_args()


def _parse_mapping(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            return None
    return value if isinstance(value, dict) else None


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _prompt_from_row(row: dict[str, str]) -> str:
    for key in PROMPT_KEYS:
        text = _clean(row.get(key))
        if text:
            mapping = _parse_mapping(text)
            if mapping:
                for inner_key in PROMPT_KEYS:
                    inner = _clean(mapping.get(inner_key))
                    if inner:
                        return inner
                for value in mapping.values():
                    inner = _clean(value)
                    if inner:
                        return inner
            return text
    for value in row.values():
        mapping = _parse_mapping(value or "")
        if mapping:
            for key in PROMPT_KEYS:
                text = _clean(mapping.get(key))
                if text:
                    return text
    return ""


def _download(args: argparse.Namespace) -> None:
    include = args.include or f"{args.subset.rstrip('/')}/*"
    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = local_dir / ".modelscope_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MODELSCOPE_CACHE", str(cache_dir.resolve()))

    try:
        from modelscope import dataset_snapshot_download

        dataset_snapshot_download(
            dataset_id=args.dataset_id,
            local_dir=str(local_dir),
            allow_file_pattern=include,
        )
        return
    except Exception as py_exc:  # noqa: BLE001 - fall back to the CLI for any SDK/runtime failure.
        print(f"[prepare_diffsynth_example] Python ModelScope download failed: {py_exc}", file=sys.stderr)

    cmd = [
        "modelscope",
        "download",
        "--dataset",
        args.dataset_id,
        "--include",
        include,
        "--local_dir",
        str(local_dir),
    ]
    print("[prepare_diffsynth_example] running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _find_metadata(local_dir: Path, subset: str) -> Path:
    subset_dir = local_dir / subset
    candidates = []
    if subset_dir.exists():
        candidates.extend(subset_dir.rglob("metadata.csv"))
    candidates.extend(local_dir.rglob("metadata.csv"))
    if not candidates:
        raise FileNotFoundError(f"No metadata.csv found under {local_dir}. Did the dataset download succeed?")
    return min(candidates, key=lambda p: (len(p.parts), str(p)))


def _extract_prompts(metadata_path: Path, output_path: Path, max_rows: int) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    prompts: list[str] = []
    with metadata_path.open(newline="", encoding="utf-8-sig", errors="replace") as f:
        for row in csv.DictReader(f):
            prompt = _prompt_from_row(row)
            if prompt and prompt not in seen:
                prompts.append(prompt)
                seen.add(prompt)
            if len(prompts) >= max_rows:
                break
    if not prompts:
        raise RuntimeError(f"No prompts could be extracted from {metadata_path}")
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt"])
        writer.writeheader()
        for prompt in prompts:
            writer.writerow({"prompt": prompt})
    return len(prompts)


def main() -> None:
    args = parse_args()
    if not args.no_download:
        _download(args)
    local_dir = Path(args.local_dir)
    metadata_path = _find_metadata(local_dir, args.subset)
    n = _extract_prompts(metadata_path, Path(args.output), args.max_rows)
    print(f"[prepare_diffsynth_example] metadata: {metadata_path}", flush=True)
    print(f"[prepare_diffsynth_example] wrote {n} prompts -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
