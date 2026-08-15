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
import io
import json
import os
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

PROMPT_KEYS = (
    "prompt",
    "instruction",
    "instructions",
    "edit_instruction",
    "editing_instruction",
    "edit_prompt",
    "target_prompt",
    "edited_prompt_list",
    "caption",
    "text",
)
SOURCE_KEYS = (
    "src_img",
    "source_image",
    "input_image",
    "original_image",
    "before_image",
    "image",
    "src_image",
    "source",
)
TARGET_KEYS = (
    "edited_img",
    "edited_image",
    "target_image",
    "output_image",
    "after_image",
    "result_image",
    "edit_image",
    "target",
)
TASK_KEYS = ("task", "category", "edit_type", "type")
QUALITY_KEYS = ("o_score", "quality", "score", "edit_score", "overall_score", "alignment_score")
ID_KEYS = ("omni_edit_id", "uid", "id", "sample_id")

GLOBAL_TASK_HINTS = ("style", "env", "background", "weather", "lighting", "tone", "scene")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare OmniEdit-style metadata for DanceOPD.")
    parser.add_argument("--input", required=True, help="Local .csv/.json/.jsonl file or Hugging Face dataset ID.")
    parser.add_argument("--split", default="train", help="HF dataset split when --input is a dataset ID.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", choices=["prompts", "sft_pairs", "danceopd"], default="prompts")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--no-streaming", action="store_true",
        help="Download a complete Hugging Face split before iterating. Streaming is the safe default.",
    )
    parser.add_argument(
        "--image-dir", default=None,
        help="Directory for materialized image pairs. Default: '<output stem>_images' beside the CSV.",
    )
    parser.add_argument(
        "--route-mode", choices=["raw", "local_global"], default="local_global",
        help="For --format danceopd, preserve raw tasks or map them to local_edit/global_edit routes.",
    )
    parser.add_argument(
        "--task-map", default=None,
        help="Optional comma-separated overrides such as style=global_edit,swap=local_edit.",
    )
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
    lower = {str(k).lower(): k for k in row}
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


def _read_hf(dataset_id: str, split: str, *, streaming: bool = True) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("Hugging Face dataset input requires `pip install datasets`.") from exc
    dataset = load_dataset(dataset_id, split=split, streaming=streaming)
    for row in dataset:
        yield dict(row)


def iter_rows(input_value: str, split: str, *, streaming: bool = True) -> Iterable[dict[str, Any]]:
    path = Path(input_value)
    if path.exists():
        yield from _read_local(path)
    else:
        yield from _read_hf(input_value, split, streaming=streaming)


def make_uid(*parts: str) -> str:
    text = "||".join(parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _parse_task_map(text: str | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in (text or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"--task-map entries must use raw_task=route: {item!r}")
        raw, route = (x.strip() for x in item.split("=", 1))
        if route not in {"local_edit", "global_edit"}:
            raise ValueError(f"Unsupported DanceOPD route {route!r} in --task-map")
        result[raw.lower()] = route
    return result


def _route_task(raw_task: str, mode: str, overrides: dict[str, str]) -> str:
    task = _clean(raw_task).lower() or "edit"
    if mode == "raw":
        return task
    if task in overrides:
        return overrides[task]
    return "global_edit" if any(hint in task for hint in GLOBAL_TASK_HINTS) else "local_edit"


def _image_payload(value: Any) -> Any:
    """Resolve a datasets Image/PIL/path/URL value to something PIL can open."""
    if value is None:
        return None
    if hasattr(value, "save") and hasattr(value, "convert"):
        return value
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return io.BytesIO(value["bytes"])
        for key in ("path", "src", "url"):
            if value.get(key):
                return _image_payload(value[key])
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith(("http://", "https://")):
        with urllib.request.urlopen(text, timeout=60) as response:
            return io.BytesIO(response.read())
    path = Path(text).expanduser()
    return path if path.exists() else None


def _materialize_image(value: Any, destination: Path) -> str:
    payload = _image_payload(value)
    if payload is None:
        return ""
    destination.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    if hasattr(payload, "save") and hasattr(payload, "convert"):
        image = payload
    else:
        image = Image.open(payload)
    image.convert("RGB").save(destination, format="JPEG", quality=95)
    return str(destination)


def _first_value(row: dict[str, Any], keys: Iterable[str], explicit: str | None = None) -> Any:
    if explicit:
        return row.get(explicit)
    lower = {str(key).lower(): key for key in row}
    for key in keys:
        actual = key if key in row else lower.get(key.lower())
        if actual is not None and row.get(actual) is not None:
            return row.get(actual)
    return None


def _relative_to_csv(path: str, output: Path) -> str:
    return os.path.relpath(path, start=output.resolve().parent) if path else ""


def main() -> None:
    args = parse_args()
    task_allow = {x.strip() for x in args.task_include.split(",") if x.strip()} if args.task_include else None
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.image_dir) if args.image_dir else output.with_suffix("").with_name(output.stem + "_images")
    task_map = _parse_task_map(args.task_map)

    seen_prompts: set[str] = set()
    n_scan = n_write = n_skip = 0
    if args.format == "prompts":
        fieldnames = ["prompt"]
    elif args.format == "sft_pairs":
        fieldnames = ["uid", "source_image", "edited_image", "prompt", "task", "caption_dict"]
    else:
        fieldnames = ["uid", "task", "raw_task", "prompt", "source_image", "target_image", "caption_dict"]

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in iter_rows(args.input, args.split, streaming=not args.no_streaming):
            n_scan += 1
            prompt = _first(row, PROMPT_KEYS, args.prompt_column)
            source_value = _first_value(row, SOURCE_KEYS, args.source_column)
            target_value = _first_value(row, TARGET_KEYS, args.target_column)
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
                raw_uid = _first(row, ID_KEYS)
                uid = raw_uid or make_uid(task, prompt, str(n_scan))
                source = _relative_to_csv(
                    _materialize_image(source_value, image_dir / "source" / f"{uid}.jpg"), output
                )
                target = _relative_to_csv(
                    _materialize_image(target_value, image_dir / "target" / f"{uid}.jpg"), output
                )
                if not source or not target:
                    n_skip += 1
                    continue
                if args.format == "sft_pairs":
                    record = {
                        "uid": uid,
                        "source_image": source,
                        "edited_image": target,
                        "prompt": prompt,
                        "task": task,
                        "caption_dict": json.dumps({"prompt": prompt, "task": task}, ensure_ascii=False),
                    }
                else:
                    route = _route_task(task, args.route_mode, task_map)
                    record = {
                        "uid": uid,
                        "task": route,
                        "raw_task": task,
                        "prompt": prompt,
                        "source_image": source,
                        "target_image": target,
                        "caption_dict": json.dumps(
                            {"prompt": prompt, "task": route, "raw_task": task}, ensure_ascii=False
                        ),
                    }
                writer.writerow(record)
            n_write += 1
            if args.max_rows is not None and n_write >= args.max_rows:
                break

    if n_write == 0:
        raise RuntimeError(
            "No usable rows were written. Check the dataset split/column names and image decoding dependencies."
        )
    print(f"[prepare_omniedit] scanned={n_scan} wrote={n_write} skipped={n_skip} -> {output}", flush=True)


if __name__ == "__main__":
    main()
