"""CSV prompt loader."""
from __future__ import annotations

import ast
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Sample:
    prompt: str
    task: str = "default"
    source_image: str | None = None
    target_image: str | None = None


@dataclass
class PromptCSV:
    samples: list[Sample]
    allow_default_fallback: bool = False

    @classmethod
    def from_config(cls, cfg) -> PromptCSV:
        csv_path = cfg.data.prompts_csv
        prompt_column = cfg.data.prompt_column
        caption_dict_column = cfg.data.caption_dict_column
        caption_dict_keys = list(cfg.data.caption_dict_keys or [])
        samples = load_samples(
            csv_path,
            prompt_column,
            caption_dict_column,
            caption_dict_keys,
            task_column=cfg.data.get("task_column", "task"),
            source_image_column=cfg.data.get("source_image_column", "source_image"),
            target_image_column=cfg.data.get("target_image_column", "target_image"),
        )
        if not samples:
            raise ValueError(f"No prompts found in CSV: {csv_path}")
        return cls(samples=samples, allow_default_fallback=bool(cfg.data.get("allow_default_fallback", False)))

    @property
    def prompts(self) -> list[str]:
        """Backward-compatible prompt view."""
        return [s.prompt for s in self.samples]

    def for_task(self, task: str) -> list[Sample]:
        exact = [s for s in self.samples if s.task == task]
        if exact:
            return exact
        if self.allow_default_fallback:
            wildcard = [s for s in self.samples if s.task in {"", "default", "*"}]
            if wildcard:
                return wildcard
        raise ValueError(f"No data rows for route dataset/task={task!r}")

    def sample(self, task: str) -> Sample:
        return random.choice(self.for_task(task))

    def validate_routes(self, routes, *, require_target_for_edit: bool = False) -> None:
        """Validate every configured bucket before any large model is loaded."""
        errors: list[str] = []
        for route in routes:
            try:
                rows = self.for_task(route.dataset)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if route.requires_source_image:
                missing_source = [row.prompt for row in rows if not row.source_image]
                missing_source_file = [
                    row.source_image for row in rows
                    if row.source_image and not Path(row.source_image).is_file()
                ]
                if missing_source:
                    errors.append(
                        f"route {route.dataset!r} has {len(missing_source)} rows without source_image"
                    )
                if missing_source_file:
                    errors.append(
                        f"route {route.dataset!r} has {len(missing_source_file)} missing source files; "
                        f"first={missing_source_file[0]!r}"
                    )
                if require_target_for_edit:
                    missing_target = [row.prompt for row in rows if not row.target_image]
                    missing_target_file = [
                        row.target_image for row in rows
                        if row.target_image and not Path(row.target_image).is_file()
                    ]
                    if missing_target:
                        errors.append(
                            f"offpolicy route {route.dataset!r} has {len(missing_target)} rows without target_image"
                        )
                    if missing_target_file:
                        errors.append(
                            f"offpolicy route {route.dataset!r} has {len(missing_target_file)} missing target files; "
                            f"first={missing_target_file[0]!r}"
                        )
        if errors:
            raise ValueError("Data preflight failed:\n- " + "\n- ".join(errors))


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


def _clean_prompt(text: Any) -> str:
    return " ".join(str(text or "").split())


def load_prompts(
    csv_path: str,
    prompt_column: str = "prompt",
    caption_dict_column: str = "caption_dict",
    caption_dict_keys: list[str] | None = None,
) -> list[str]:
    return [s.prompt for s in load_samples(csv_path, prompt_column, caption_dict_column, caption_dict_keys)]


def load_samples(
    csv_path: str,
    prompt_column: str = "prompt",
    caption_dict_column: str = "caption_dict",
    caption_dict_keys: list[str] | None = None,
    *,
    task_column: str = "task",
    source_image_column: str = "source_image",
    target_image_column: str = "target_image",
) -> list[Sample]:
    caption_dict_keys = caption_dict_keys or ["prompt", "caption"]
    samples: list[Sample] = []
    csv_root = Path(csv_path).resolve().parent

    def image_path(value: Any) -> str | None:
        text = str(value or "").strip()
        if not text:
            return None
        path = Path(text).expanduser()
        return str(path if path.is_absolute() else csv_root / path)
    csv.field_size_limit(2**31 - 1)
    with open(csv_path, newline="", encoding="utf-8-sig", errors="replace") as f:
        for row in csv.DictReader(f):
            prompt = _clean_prompt(row.get(prompt_column))
            if not prompt and caption_dict_column:
                mapping = _parse_mapping(row.get(caption_dict_column) or "")
                if mapping:
                    for key in caption_dict_keys:
                        prompt = _clean_prompt(mapping.get(key))
                        if prompt:
                            break
                    if not prompt:
                        prompt = next((_clean_prompt(v) for v in mapping.values() if _clean_prompt(v)), "")
            if prompt:
                samples.append(
                    Sample(
                        prompt=prompt,
                        task=_clean_prompt(row.get(task_column)) or "default",
                        source_image=image_path(row.get(source_image_column)),
                        target_image=image_path(row.get(target_image_column)),
                    )
                )
    return samples
