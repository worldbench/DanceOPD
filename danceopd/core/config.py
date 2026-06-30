"""Configuration helpers for the public DanceOPD trainer."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "backend": "sd35",
    "model": {
        "pretrained_model": None,
        "model_paths": None,
        "model_id_with_origin_paths": None,
        "tokenizer_path": None,
    },
    "student": {
        "init": "base",
        "lora_rank": 128,
        "lora_alpha": 128,
        "lora_target_modules": None,
    },
    "teachers": [
        {
            "name": "default",
            "base_ckpt": None,
            "lora_dir": None,
            "weight": 1.0,
        }
    ],
    "routing": {
        "routes": [
            {"teacher": "default", "weight": 1.0},
        ]
    },
    "data": {
        "prompts_csv": None,
        "prompt_column": "prompt",
        "caption_dict_column": "caption_dict",
        "caption_dict_keys": ["prompt", "caption"],
    },
    "training": {
        "method": "danceopd",
        "output_dir": None,
        "resolution": 1024,
        "height": None,
        "width": None,
        "rollout_steps": 16,
        "k": 1,
        "query_bias": "low_t",
        "lr": 2.0e-4,
        "weight_decay": 0.0,
        "grad_accum": 4,
        "max_train_steps": 3000,
        "save_steps": 300,
        "mixed_precision": "bf16",
        "gradient_clip": 1.0,
        "max_sequence_length": 256,
        "log_every": 20,
    },
}


class Config(dict):
    """Small dict subclass with dot access and dotted-key lookup."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def get_dotted(self, dotted_key: str, default: Any = None) -> Any:
        cur: Any = self
        for part in dotted_key.split("."):
            if isinstance(cur, Mapping) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur


def _as_config(value: Any) -> Any:
    if isinstance(value, Mapping):
        return Config({k: _as_config(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_as_config(v) for v in value]
    return value


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(config_file: str | None) -> Config:
    raw: dict[str, Any] = {}
    if config_file:
        with open(config_file, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, Mapping):
            raise ValueError(f"Config must be a mapping: {config_file}")
        raw = dict(loaded)
    merged = _deep_merge(DEFAULT_CONFIG, raw)
    return _as_config(merged)


def _step(container: Any, part: str) -> Any:
    if isinstance(container, list):
        return container[int(part)]
    if isinstance(container, Mapping):
        return container[part]
    raise TypeError(f"Cannot descend into {type(container).__name__} with key {part!r}")


def _set_step(container: Any, part: str, value: Any) -> None:
    if isinstance(container, list):
        container[int(part)] = value
    elif isinstance(container, dict):
        container[part] = value
    else:
        raise TypeError(f"Cannot set {part!r} on {type(container).__name__}")


def apply_overrides(cfg: Config, overrides: list[str] | None) -> Config:
    """Apply command-line overrides of the form `a.b=value`.

    List entries can be addressed with numeric components, e.g.
    `teachers.0.lora_dir=...`.
    """
    if not overrides:
        return cfg
    raw = copy.deepcopy(dict(cfg))
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must use key=value form: {item}")
        key, text_value = item.split("=", 1)
        value = yaml.safe_load(text_value)
        cur: Any = raw
        parts = key.split(".")
        for part in parts[:-1]:
            cur = _step(cur, part)
        _set_step(cur, parts[-1], value)
    return _as_config(raw)


def require_value(cfg: Config, dotted_key: str) -> Any:
    value = cfg.get_dotted(dotted_key)
    if value in (None, ""):
        raise ValueError(f"Missing required config value: {dotted_key}")
    return value


def teacher_map(cfg: Config) -> dict[str, Config]:
    teachers = cfg.get("teachers", [])
    out: dict[str, Config] = {}
    for teacher in teachers:
        name = teacher.get("name")
        if not name:
            raise ValueError("Every teacher entry must have a name.")
        out[str(name)] = teacher
    return out


def print_config_summary(cfg: Config) -> str:
    routes = cfg.get_dotted("routing.routes", [])
    route_desc = ", ".join(f"{r.get('teacher')}:{r.get('weight', 1.0)}" for r in routes)
    return (
        f"backend={cfg.backend} method={cfg.training.method} "
        f"K={cfg.training.k} rollout={cfg.training.rollout_steps} "
        f"bias={cfg.training.query_bias} routes=[{route_desc}] "
        f"lr={cfg.training.lr} grad_accum={cfg.training.grad_accum} "
        f"steps={cfg.training.max_train_steps}"
    )
