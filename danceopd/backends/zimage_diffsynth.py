"""Z-Image / DiffSynth backend for DanceOPD.

This backend keeps the same public DanceOPD core as the SD3.5 backend, but uses
DiffSynth's ZImagePipeline and DiT velocity function. DiffSynth releases may
rename small helper methods; the adapter therefore uses conservative fallbacks
and raises actionable errors when a local install exposes a different API.
"""
from __future__ import annotations

import copy
import os
from typing import Any

import torch
from safetensors.torch import save_file

from danceopd.backends.base import DanceOPDBackend
from danceopd.backends.teacher import compose_teacher, load_state_dict_file
from danceopd.core.checkpoint import ensure_output_dir, step_dir
from danceopd.core.config import teacher_map
from danceopd.core.loss import velocity_mse
from danceopd.core.rollout import FlowState
from danceopd.core.timestep import sample_query_indices


def _dtype(name: str):
    name = (name or "bf16").lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def _split_csv(text: str | None) -> list[str]:
    return [x.strip() for x in str(text or "").split(",") if x.strip()]


class ZImageBackend(DanceOPDBackend):
    """DanceOPD backend for Z-Image through DiffSynth-Studio."""

    def __init__(self, cfg, accelerator):
        self.cfg = cfg
        self.accelerator = accelerator
        self.device = accelerator.device
        self.torch_dtype = _dtype(cfg.training.mixed_precision)
        self.pipe = None
        self.student = None
        self.optimizer = None
        self.teachers: dict[str, Any] = {}
        self.trainable_params = []

    def prepare(self) -> None:
        try:
            from diffsynth.pipelines.z_image import ModelConfig, ZImagePipeline
        except Exception as exc:
            raise RuntimeError(
                "Z-Image backend requires DiffSynth-Studio with `diffsynth.pipelines.z_image`. "
                "Install DiffSynth first, then run with the zimage extra."
            ) from exc
        from peft import LoraConfig

        model_configs = self._model_configs(ModelConfig)
        tokenizer_config = self._make_model_config(ModelConfig, self.cfg.model.tokenizer_path) if self.cfg.model.tokenizer_path else None
        kwargs = {"torch_dtype": self.torch_dtype, "device": self.device, "model_configs": model_configs}
        if tokenizer_config is not None:
            kwargs["tokenizer_config"] = tokenizer_config
        self.pipe = ZImagePipeline.from_pretrained(**kwargs)
        if not hasattr(self.pipe, "dit"):
            raise RuntimeError("Loaded ZImagePipeline has no `dit` module; check DiffSynth version.")
        if not hasattr(self.pipe, "model_fn"):
            raise RuntimeError("Loaded ZImagePipeline has no `model_fn`; check DiffSynth version.")

        # Keep a pristine DiT copy for teacher construction. Teacher modules
        # must not inherit the student's freshly initialized training LoRA;
        # otherwise a teacher LoRA would be loaded on top of another LoRA.
        teacher_template = copy.deepcopy(self.pipe.dit).to("cpu").eval().requires_grad_(False)

        student = self.pipe.dit
        student.requires_grad_(False)
        if self.cfg.student.init and self.cfg.student.init not in {"base", "none"}:
            init_path = self._resolve_student_init(self.cfg.student.init)
            if init_path:
                sd = load_state_dict_file(init_path)
                missing, unexpected = student.load_state_dict(sd, strict=False)
                self._log(f"student warm-start loaded tensors={len(sd)} missing={len(missing)} unexpected={len(unexpected)}")

        target_modules = self.cfg.student.lora_target_modules or ["to_q", "to_k", "to_v", "to_out.0", "w1", "w2", "w3"]
        rank = int(self.cfg.student.lora_rank)
        alpha = int(self.cfg.student.lora_alpha or rank)
        lora_cfg = LoraConfig(r=rank, lora_alpha=alpha, init_lora_weights="gaussian", target_modules=target_modules)
        if hasattr(student, "add_adapter"):
            student.add_adapter(lora_cfg)
        else:
            raise RuntimeError(
                "The loaded Z-Image DiT does not expose PEFT `add_adapter`. "
                "Use a DiffSynth build with PEFT support or wrap the DiT before training."
            )

        self.teachers = {}
        for name, tcfg in teacher_map(self.cfg).items():
            teacher = copy.deepcopy(teacher_template)
            teacher = compose_teacher(teacher, tcfg, label=f"teacher={name}", log=self._log, device=self.device)
            self.teachers[name] = teacher

        self.trainable_params = [p for p in student.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            self.trainable_params,
            lr=float(self.cfg.training.lr),
            weight_decay=float(self.cfg.training.weight_decay),
        )
        self.student, self.optimizer = self.accelerator.prepare(student, self.optimizer)
        self.pipe.dit = self.accelerator.unwrap_model(self.student)
        ensure_output_dir(self.cfg.training.output_dir)
        n_train = sum(p.numel() for p in self.trainable_params)
        self._log(f"student LoRA rank={rank} trainable_params={n_train/1e6:.2f}M")

    def _make_model_config(self, ModelConfig, spec: str):
        spec = str(spec)
        if ":" in spec and not os.path.exists(spec):
            model_id, pattern = spec.split(":", 1)
            return ModelConfig(model_id=model_id, origin_file_pattern=pattern)
        return ModelConfig(spec)

    def _model_configs(self, ModelConfig):
        if self.cfg.model.model_id_with_origin_paths:
            return [self._make_model_config(ModelConfig, spec) for spec in _split_csv(self.cfg.model.model_id_with_origin_paths)]
        if self.cfg.model.model_paths:
            return [self._make_model_config(ModelConfig, path) for path in _split_csv(self.cfg.model.model_paths)]
        raise ValueError("Z-Image requires model.model_paths or model.model_id_with_origin_paths.")

    def _resolve_student_init(self, value: str) -> str | None:
        if value in {"teacher", "teacher_base"}:
            teachers = list(teacher_map(self.cfg).values())
            return teachers[0].get("base_ckpt") if teachers else None
        return value

    @torch.no_grad()
    def _encode_prompt(self, prompt: str):
        pipe = self.pipe
        if hasattr(pipe, "encode_prompt"):
            return pipe.encode_prompt(prompt)
        if hasattr(pipe, "prompter"):
            prompter = pipe.prompter
            for method_name in ("encode_prompt", "encode", "__call__"):
                method = getattr(prompter, method_name, None)
                if method is not None:
                    try:
                        out = method(prompt, device=self.device)
                    except TypeError:
                        out = method(prompt)
                    if isinstance(out, tuple):
                        out = out[0]
                    return out.to(self.device, dtype=self.torch_dtype) if hasattr(out, "to") else out
        raise RuntimeError(
            "Could not encode text with this ZImagePipeline. Add a small adapter in "
            "ZImageBackend._encode_prompt for your DiffSynth version."
        )

    def _latent_shape(self) -> tuple[int, int, int, int]:
        height = int(self.cfg.training.height or self.cfg.training.resolution)
        width = int(self.cfg.training.width or self.cfg.training.resolution)
        latent_channels = int(self.cfg.training.get("latent_channels", 16))
        downsample = int(self.cfg.training.get("latent_downsample_factor", 8))
        return (1, latent_channels, height // downsample, width // downsample)

    def _velocity(self, model, latents: torch.Tensor, timestep: torch.Tensor, prompt_embeds):
        return self.pipe.model_fn(
            model,
            latents=latents.to(self.torch_dtype),
            timestep=timestep.reshape(1).to(self.device, dtype=self.torch_dtype),
            prompt_embeds=prompt_embeds,
            image_latents=None,
            use_gradient_checkpointing=bool(self.cfg.training.get("use_gradient_checkpointing", True)),
        )

    @torch.no_grad()
    def _timesteps(self):
        scheduler = self.pipe.scheduler
        steps = int(self.cfg.training.rollout_steps)
        if hasattr(scheduler, "set_timesteps"):
            try:
                scheduler.set_timesteps(steps, device=self.device)
            except TypeError:
                scheduler.set_timesteps(steps)
        if not hasattr(scheduler, "timesteps"):
            raise RuntimeError("Z-Image scheduler has no timesteps after set_timesteps().")
        return scheduler.timesteps.to(self.device)

    @torch.no_grad()
    def _rollout(self, model, prompt_embeds) -> tuple[list[FlowState], torch.Tensor]:
        timesteps = self._timesteps()
        steps = int(self.cfg.training.rollout_steps)
        idx = [int(round(i * (len(timesteps) - 1) / steps)) for i in range(steps)]
        latents = torch.randn(self._latent_shape(), device=self.device, dtype=self.torch_dtype)
        trajectory: list[FlowState] = []
        for pos, ts_index in enumerate(idx):
            t_cur = timesteps[ts_index]
            if pos + 1 < len(idx):
                t_next = timesteps[idx[pos + 1]]
            else:
                t_next = torch.zeros_like(t_cur)
            v = self._velocity(model, latents, t_cur, prompt_embeds)
            trajectory.append(FlowState(latents=latents.detach().clone(), timestep=t_cur.detach().clone()))
            dt = (t_next - t_cur) / 1000.0
            latents = (latents + dt * v).to(self.torch_dtype)
        return trajectory, latents.detach()

    def _renoise(self, x0: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        scheduler = self.pipe.scheduler
        noise = torch.randn_like(x0)
        if hasattr(scheduler, "add_noise"):
            return scheduler.add_noise(x0, noise, timestep.reshape(1)).to(self.torch_dtype)
        t = (timestep.float() / 1000.0).reshape(1, 1, 1, 1).to(self.device)
        return ((1.0 - t) * x0.float() + t * noise.float()).to(self.torch_dtype)

    def compute_loss(self, prompt: str, route) -> torch.Tensor:
        prompt_embeds = self._encode_prompt(prompt)
        teacher = self.teachers[route.teacher]
        method = str(self.cfg.training.method).lower()
        raw_student = self.accelerator.unwrap_model(self.student)

        if method == "offpolicy":
            _, x0 = self._rollout(teacher, prompt_embeds)
            timesteps = self._timesteps()
            indices = sample_query_indices(len(timesteps), int(self.cfg.training.k), str(self.cfg.training.query_bias))
            states = [FlowState(latents=self._renoise(x0, timesteps[i]).detach(), timestep=timesteps[i].detach()) for i in indices]
        else:
            trajectory, _ = self._rollout(raw_student, prompt_embeds)
            indices = sample_query_indices(len(trajectory), int(self.cfg.training.k), str(self.cfg.training.query_bias))
            states = [trajectory[i] for i in indices]

        total = None
        for state in states:
            z = state.latents.to(self.device, dtype=self.torch_dtype)
            t = state.timestep.to(self.device)
            with torch.no_grad():
                target = self._velocity(teacher, z, t, prompt_embeds)
            pred = self._velocity(self.student, z, t, prompt_embeds)
            loss = velocity_mse(pred, target) / len(states)
            total = loss if total is None else total + loss
        return total

    def backward(self, loss: torch.Tensor) -> None:
        self.accelerator.backward(loss)

    def optimizer_step(self) -> None:
        clip = float(self.cfg.training.gradient_clip or 0.0)
        if clip > 0:
            self.accelerator.clip_grad_norm_(self.trainable_params, clip)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def save(self, step: int) -> None:
        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            return
        out = step_dir(self.cfg.training.output_dir, step)
        os.makedirs(out, exist_ok=True)
        raw = self.accelerator.unwrap_model(self.student)
        state = {k: v.detach().cpu().contiguous() for k, v in raw.state_dict().items() if "lora_" in k}
        save_file(state, os.path.join(out, "adapter_model.safetensors"))
        print(f"[DanceOPD][zimage] saved {out}", flush=True)

    def _log(self, msg: str) -> None:
        if self.accelerator.is_main_process:
            print(f"[DanceOPD][zimage] {msg}", flush=True)
