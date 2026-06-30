#!/usr/bin/env python3
"""Convert OmniEdit-style edit metadata into public DanceOPD CSV formats.

The script accepts either local CSV/JSON/JSONL metadata or a Hugging Face dataset
ID such as TIGER-Lab/OmniEdit-Filtered-1.2M. It is intentionally schema-tolerant:
common column aliases are auto-detected unless explicit column names are passed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

PROMPT_KEYS = (
    "prompt",
    "instruction",
    "instructions",
    "edit_instruction",
    "editing_instruction",
    "edit_prompt",
    "target_prompt",
    "caption",
    "text",
)
SOURCE_KEYS = (
    "source_image",
    "input_image",
    "original_image",
    "before_image",
    "image",
    "src_image",
    "source",
)
TARGET_KEYS = (
    "edited_image",
    "target_image",
    "output_image",
    "after_image",
    "result_image",
    "edit_image",
    "target",
)
TASK_KEYS = ("task", "category", "edit_type", "type")
QUALITY_KEYS = ("quality", "score", "edit_score", "overall_score", "alignment_score")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare OmniEdit-style metadata for DanceOPD.")
    parser.add_argument("--input", required=True, help="Local .csv/.json/.jsonl file or Hugging Face dataset ID.")
    parser.add_argument("--split", default="train", help="HF dataset split when --input is a dataset ID.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", choices=["prompts", "sft_pairs"], default="prompts")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--task-include", default=None, help="Comma-separated task/category allow-list.")
    parser.add_argument("--min-quality", type=float, default=None)
    parser.add_argument("--prompt-column", default=None)
    parser.add_argument("--source-column", default=None)
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--task-column", default=None)
    parser.add_argument("--quality-column", default=None)
    return parser.parse_args()


def _clean(value: Any) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    if isinstance(value, dict):
        for key in PROMPT_KEYS:
            if key in value:
                return _clean(value[key])
        return json.dumps(value, ensure_ascii=False)
    return " ".join(str(value or "").split())


def _first(row: dict[str, Any], keys: Iterable[str], explicit: str | None = None) -> str:
    if explicit:
        return _clean(row.get(explicit))
    lower = {str(k).lower(): k for k in row.keys()}
    for key in keys:
        if key in row:
            return _clean(row.get(key))
        if key.lower() in lower:
            return _clean(row.get(lower[key.lower()]))
    return ""


def _read_local(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig", errors="replace") as f:
            yield from csv.DictReader(f)
        return
    if suffix == ".jsonl":
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ("items", "data", "entries", "rows"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        if not isinstance(data, list):
            raise ValueError(f"Unsupported JSON shape in {path}")
        for item in data:
            if isinstance(item, dict):
                yield item
        return
    raise ValueError(f"Unsupported local metadata format: {path}")


def _read_hf(dataset_id: str, split: str) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Hugging Face dataset input requires `pip install datasets`.") from exc
    dataset = load_dataset(dataset_id, split=split)
    for row in dataset:
        yield dict(row)


def iter_rows(input_value: str, split: str) -> Iterable[dict[str, Any]]:
    path = Path(input_value)
    if path.exists():
        yield from _read_local(path)
    else:
        yield from _read_hf(input_value, split)


def make_uid(*parts: str) -> str:
    text = "||".join(parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def main() -> None:
    args = parse_args()
    task_allow = {x.strip() for x in args.task_include.split(",") if x.strip()} if args.task_include else None
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    seen_prompts: set[str] = set()
    n_scan = n_write = n_skip = 0
    fieldnames = ["prompt"] if args.format == "prompts" else ["uid", "source_image", "edited_image", "prompt", "task", "caption_dict"]

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in iter_rows(args.input, args.split):
            n_scan += 1
            prompt = _first(row, PROMPT_KEYS, args.prompt_column)
            source = _first(row, SOURCE_KEYS, args.source_column)
            target = _first(row, TARGET_KEYS, args.target_column)
            task = _first(row, TASK_KEYS, args.task_column)
            quality = _first(row, QUALITY_KEYS, args.quality_column)

            if task_allow and task not in task_allow:
                n_skip += 1
                continue
            if args.min_quality is not None and quality:
                try:
                    if float(quality) < args.min_quality:
                        n_skip += 1
                        continue
                except ValueError:
                    pass
            if not prompt:
                n_skip += 1
                continue

            if args.format == "prompts":
                if prompt in seen_prompts:
                    n_skip += 1
                    continue
                seen_prompts.add(prompt)
                writer.writerow({"prompt": prompt})
            else:
                if not source or not target:
                    n_skip += 1
                    continue
                uid = make_uid(source, target, prompt)
                writer.writerow(
                    {
                        "uid": uid,
                        "source_image": source,
                        "edited_image": target,
                        "prompt": prompt,
                        "task": task,
                        "caption_dict": json.dumps({"prompt": prompt, "task": task}, ensure_ascii=False),
                    }
                )
            n_write += 1
            if args.max_rows is not None and n_write >= args.max_rows:
                break

    print(f"[prepare_omniedit] scanned={n_scan} wrote={n_write} skipped={n_skip} -> {output}", flush=True)


if __name__ == "__main__":
    main()
