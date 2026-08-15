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
        "init": "auto",
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
        "granularity": "optimizer_step",
        # Optional DiffusionOPD/FlowOPD G=M update. Entries select configured
        # routes by dataset name (or teacher name when unambiguous).
        "accumulation_groups": None,
        "routes": [
            {"teacher": "default", "dataset": "default", "weight": 1.0},
        ]
    },
    "data": {
        "prompts_csv": None,
        "prompt_column": "prompt",
        "caption_dict_column": "caption_dict",
        "caption_dict_keys": ["prompt", "caption"],
        "task_column": "task",
        "source_image_column": "source_image",
        "target_image_column": "target_image",
        "allow_default_fallback": False,
    },
    "training": {
        "method": "danceopd",
        # CFG fields are ordinary train-time options, not a separate method.
        # Defaults match the ordinary conditional teacher/student fields.
        "teacher_cfg_scale": 1.0,
        "student_cfg_scale": 1.0,
        "flowopd_noise_level": 0.7,
        "flowopd_group_size": 16,
        "flowopd_k_states": None,
        "flowopd_query_bias": "uniform",
        "flowopd_clip_range": 1.0e-4,
        "flowopd_adv_clip_max": 5.0,
        "flowopd_kl_scale": -1.0,
        "flowopd_anchor_beta": 0.0,
        "flowopd_mar_teacher": None,
        "output_dir": None,
        "resume_from": None,
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

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = _as_config(value)

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
        if str(name) in out:
            raise ValueError(f"Duplicate teacher name: {name!r}")
        out[str(name)] = teacher
    return out


def validate_config(cfg: Config) -> None:
    """Fail early on cross-field errors that otherwise waste a GPU launch."""
    teachers = teacher_map(cfg)
    routes = cfg.get_dotted("routing.routes", [])
    if not routes:
        raise ValueError("routing.routes must not be empty")
    unknown = [r.get("teacher") for r in routes if r.get("teacher") not in teachers]
    if unknown:
        raise ValueError(f"Routes reference unknown teachers: {unknown}")
    from danceopd.core.methods import get_method

    method = get_method(cfg.training.method)
    if method.name == "danceopd" and int(cfg.training.k) < 1:
        raise ValueError("DanceOPD requires training.k >= 1")
    if str(cfg.routing.get("granularity", "optimizer_step")) != "optimizer_step":
        raise ValueError("Only optimizer_step routing is supported; mixing routes inside accumulation is unsafe")
    groups = cfg.routing.get("accumulation_groups")
    if groups:
        if method.name not in {"diffusionopd", "flowopd"}:
            raise ValueError("routing.accumulation_groups is only valid for DiffusionOPD/FlowOPD")
        known_datasets = {str(r.get("dataset", r.get("task", "default"))) for r in routes}
        known_teachers = {str(r.get("teacher")) for r in routes}
        missing = [str(g) for g in groups if str(g) not in known_datasets | known_teachers]
        if missing:
            raise ValueError(f"Unknown routing.accumulation_groups entries: {missing}")
    if int(cfg.training.get("flowopd_group_size", 16)) < 1:
        raise ValueError("training.flowopd_group_size must be >= 1")
    if int(cfg.training.get("rollout_steps", 16)) < 1:
        raise ValueError("training.rollout_steps must be >= 1")
    if int(cfg.training.get("grad_accum", 1)) < 1:
        raise ValueError("training.grad_accum must be >= 1")
    mar_teacher = cfg.training.get("flowopd_mar_teacher")
    if float(cfg.training.get("flowopd_anchor_beta", 0.0)) > 0 and mar_teacher not in teachers:
        raise ValueError("training.flowopd_anchor_beta > 0 requires a loaded flowopd_mar_teacher")
    if "cfg_absorption_scale" in cfg.training:
        raise ValueError(
            "training.cfg_absorption_scale was replaced by training.teacher_cfg_scale; "
            "set training.student_cfg_scale separately"
        )
    for key in ("teacher_cfg_scale", "student_cfg_scale"):
        scale = float(cfg.training.get(key, 1.0))
        if scale < 0:
            raise ValueError(f"training.{key} must be non-negative")


def print_config_summary(cfg: Config) -> str:
    routes = cfg.get_dotted("routing.routes", [])
    route_desc = ", ".join(
        f"{r.get('dataset', r.get('task', 'default'))}->{r.get('teacher')}:{r.get('weight', 1.0)}" for r in routes
    )
    return (
        f"backend={cfg.backend} method={cfg.training.method} "
        f"K={cfg.training.k} rollout={cfg.training.rollout_steps} "
        f"bias={cfg.training.query_bias} routes=[{route_desc}] "
        f"cfg=T{cfg.training.teacher_cfg_scale:g}/S{cfg.training.student_cfg_scale:g} "
        f"lr={cfg.training.lr} grad_accum={cfg.training.grad_accum} "
        f"steps={cfg.training.max_train_steps}"
    )
