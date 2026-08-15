"""Z-Image / DiffSynth backend for DanceOPD.

This backend keeps the same public DanceOPD core as the SD3.5 backend, but uses
DiffSynth's ZImagePipeline and DiT velocity function. DiffSynth releases may
rename small helper methods; the adapter therefore uses conservative fallbacks
and raises actionable errors when a local install exposes a different API.
"""
from __future__ import annotations

import copy
import math
import os
import sys
import types
from typing import Any

import torch
from safetensors.torch import save_file

from danceopd.backends.base import DanceOPDBackend
from danceopd.backends.teacher import compose_teacher, load_state_dict_file, resolve_checkpoint
from danceopd.core.checkpoint import ensure_output_dir, load_trainer_state, save_trainer_state, step_dir
from danceopd.core.config import teacher_map
from danceopd.core.flowopd import clipped_policy_loss, sde_step
from danceopd.core.loss import velocity_mse
from danceopd.core.methods import flowopd_query_indices, get_method, query_indices
from danceopd.core.rollout import FlowState
from danceopd.core.timestep import cfg_target, rollout_grid


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
        self._ensure_modelscope_compat()
        try:
            from diffsynth.pipelines.z_image import ModelConfig, ZImagePipeline
        except Exception as exc:
            raise RuntimeError(
                "Z-Image backend requires DiffSynth-Studio with `diffsynth.pipelines.z_image`. "
                "Install DiffSynth first, then run with the zimage extra."
            ) from exc
        from peft import LoraConfig, inject_adapter_in_model

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
        target_modules = self.cfg.student.lora_target_modules or ["to_q", "to_k", "to_v", "to_out.0", "w1", "w2", "w3"]
        rank = int(self.cfg.student.lora_rank)
        alpha = int(self.cfg.student.lora_alpha or rank)
        lora_cfg = LoraConfig(r=rank, lora_alpha=alpha, init_lora_weights="gaussian", target_modules=target_modules)
        student.requires_grad_(False)
        student = inject_adapter_in_model(lora_cfg, student)

        # DiffSynth's native trainer initializes and trains the same injected
        # adapter; loading before injection silently discards all lora_* keys.
        student_init = self._default_student_init(self.cfg.student.init)
        if student_init and student_init not in {"base", "none"}:
            init_path = self._resolve_student_init(student_init)
            if init_path:
                sd = self._map_lora_state_dict(load_state_dict_file(resolve_checkpoint(init_path)))
                missing, unexpected = student.load_state_dict(sd, strict=False)
                loaded = len([k for k in sd if "lora_" in k and k not in unexpected])
                if loaded == 0:
                    raise RuntimeError(f"Student warm-start contained no compatible LoRA tensors: {init_path}")
                self._log(f"student warm-start loaded_lora={loaded} missing={len(missing)} unexpected={len(unexpected)}")

        self.teachers = {}
        for name, tcfg in teacher_map(self.cfg).items():
            teacher = copy.deepcopy(teacher_template)
            raw_lora = None
            if tcfg.get("lora_dir"):
                resolved_lora = resolve_checkpoint(tcfg.get("lora_dir"))
                raw_lora = resolved_lora if os.path.isfile(resolved_lora) else None
            if raw_lora:
                trank = int(tcfg.get("lora_rank", rank))
                talpha = int(tcfg.get("lora_alpha", trank))
                ttargets = tcfg.get("lora_target_modules", target_modules)
                teacher = inject_adapter_in_model(
                    LoraConfig(r=trank, lora_alpha=talpha, target_modules=ttargets), teacher
                )
                state = self._map_lora_state_dict(load_state_dict_file(raw_lora))
                missing, unexpected = teacher.load_state_dict(state, strict=False)
                loaded = len([k for k in state if "lora_" in k and k not in unexpected])
                if loaded == 0:
                    raise RuntimeError(f"Teacher {name!r} contained no compatible LoRA tensors: {raw_lora}")
                self._log(f"teacher={name} raw LoRA loaded_lora={loaded} missing={len(missing)} unexpected={len(unexpected)}")
                teacher_cfg = {**tcfg, "lora_dir": None}
            else:
                teacher_cfg = tcfg
            teacher = compose_teacher(teacher, teacher_cfg, label=f"teacher={name}", log=self._log, device=self.device)
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

    @staticmethod
    def _ensure_modelscope_compat() -> None:
        """Let DiffSynth load ModelScope-style specs through HF when needed.

        DiffSynth imports ``modelscope.snapshot_download`` unconditionally even
        when all model files already live in the Hugging Face cache. A minimal
        shim keeps the public backend runnable in lean environments; a working
        real ModelScope installation always wins.
        """
        try:
            from modelscope import snapshot_download as _  # noqa: F401
            return
        except (ImportError, AttributeError):
            for key in list(sys.modules):
                if key == "modelscope" or key.startswith("modelscope."):
                    sys.modules.pop(key, None)
        from huggingface_hub import snapshot_download as hf_snapshot_download

        def snapshot_download(model_id, **kwargs):
            """Translate ModelScope's downloader keyword names to HF names."""
            allow = kwargs.pop("allow_file_pattern", None)
            ignore = kwargs.pop("ignore_file_pattern", None)
            if allow is not None:
                kwargs["allow_patterns"] = allow
            if ignore:
                kwargs["ignore_patterns"] = ignore
            return hf_snapshot_download(repo_id=model_id, **kwargs)

        shim = types.ModuleType("modelscope")
        shim.snapshot_download = snapshot_download
        sys.modules["modelscope"] = shim

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

    def _default_student_init(self, value: str | None) -> str:
        if value in {None, "", "auto"}:
            # Apache-2.0 de-distillation/training LoRA used as a runnable fallback.
            return "hf://ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v2.safetensors"
        return str(value)

    @staticmethod
    def _map_lora_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mapped = {}
        for key, value in state.items():
            key = key.removeprefix("pipe.dit.").removeprefix("dit.").removeprefix("diffusion_model.")
            key = key.replace("lora_A.weight", "lora_A.default.weight")
            key = key.replace("lora_B.weight", "lora_B.default.weight")
            mapped[key] = value
        return mapped

    @torch.no_grad()
    def _encode_prompt(self, prompt: str, edit_image=None):
        pipe = self.pipe
        # Source-image editing uses DiffSynth's Omni prompt layout even for the
        # plain Z-Image DiT. A normal T2I prompt has only one caption segment and
        # cannot be paired safely with the [source, noisy-target] latent list.
        if edit_image is None and hasattr(pipe, "encode_prompt"):
            return pipe.encode_prompt(prompt)
        if edit_image is None and hasattr(pipe, "prompter"):
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
        # Current DiffSynth-Studio keeps text encoding on a PipelineUnit rather
        # than the pipeline itself.
        for unit in getattr(pipe, "units", []):
            method = (
                getattr(unit, "encode_prompt_omni", None)
                if edit_image is not None
                else getattr(unit, "encode_prompt", None)
            )
            if method is not None:
                if hasattr(pipe, "load_models_to_device"):
                    pipe.load_models_to_device(["text_encoder"])
                kwargs = {
                    "device": self.device,
                    "max_sequence_length": int(self.cfg.training.get("max_sequence_length", 512)),
                }
                if edit_image is not None:
                    return method(pipe, prompt, edit_image, **kwargs)
                return method(pipe, prompt, **kwargs)
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

    def _conditioned_velocity(
        self, model, latents: torch.Tensor, timestep: torch.Tensor, prompt_embeds, image_latents
    ) -> torch.Tensor:
        """Run Z-Image's native multi-image path for source-conditioned editing.

        DiffSynth 2.1.2 routes ordinary Z-Image through its Turbo helper, where
        ``image_latents`` is accepted but unused. Calling the DiT's multi-image
        path directly keeps each source latent clean and marks only the final
        target latent as noisy, which is the edit-conditioning contract used by
        the Z-Image training code.
        """
        condition_latents = [x.to(self.torch_dtype).permute(1, 0, 2, 3) for x in image_latents]
        target_latents = latents.to(self.torch_dtype).permute(1, 0, 2, 3)
        model_inputs = [condition_latents + [target_latents]]
        image_noise_mask = [[0] * len(condition_latents) + [1]]
        scaled_timestep = (1000.0 - timestep.reshape(1).to(self.device, dtype=self.torch_dtype)) / 1000.0
        output = model(
            model_inputs,
            scaled_timestep,
            prompt_embeds,
            siglip_feats=[None],
            image_noise_mask=image_noise_mask,
            use_gradient_checkpointing=bool(self.cfg.training.get("use_gradient_checkpointing", True)),
        )[0]
        return -output.permute(1, 0, 2, 3)

    def _velocity(self, model, latents: torch.Tensor, timestep: torch.Tensor, prompt_embeds, image_latents=None):
        if image_latents:
            return self._conditioned_velocity(model, latents, timestep, prompt_embeds, image_latents)
        return self.pipe.model_fn(
            model,
            latents=latents.to(self.torch_dtype),
            timestep=timestep.reshape(1).to(self.device, dtype=self.torch_dtype),
            prompt_embeds=prompt_embeds,
            image_latents=image_latents,
            use_gradient_checkpointing=bool(self.cfg.training.get("use_gradient_checkpointing", True)),
        )

    def _cfg_velocity(
        self, model, latents, timestep, prompt_embeds, image_latents=None,
        *, scale: float = 1.0, uncond_embeds=None,
    ):
        cond = self._velocity(model, latents, timestep, prompt_embeds, image_latents)
        if float(scale) == 1.0:
            return cond
        if uncond_embeds is None:
            raise ValueError("CFG scale != 1 requires unconditional prompt embeddings")
        uncond = self._velocity(model, latents, timestep, uncond_embeds, image_latents)
        return cfg_target(uncond, cond, float(scale))

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
    def _rollout(
        self, model, prompt_embeds, image_latents=None, *, stochastic: bool = False,
        cfg_scale: float = 1.0, uncond_embeds=None,
    ) -> tuple[list[FlowState], torch.Tensor]:
        timesteps = self._timesteps()
        steps = int(self.cfg.training.rollout_steps)
        # Need steps+1 boundaries. The old implementation constructed only
        # `steps` points and then invented a final zero, duplicating/skipping a
        # scheduler state for 16-step runs.
        # DiffSynth may expose `steps` denoising states without the clean
        # terminal boundary. Append t=0 only when it is absent.
        if len(timesteps) == steps:
            timesteps = torch.cat([timesteps, torch.zeros_like(timesteps[:1])])
        idx = rollout_grid(len(timesteps), steps)
        latents = torch.randn(self._latent_shape(), device=self.device, dtype=self.torch_dtype)
        trajectory: list[FlowState] = []
        for pos in range(steps):
            ts_index = idx[pos]
            t_cur = timesteps[ts_index]
            t_next = timesteps[idx[pos + 1]]
            v = self._cfg_velocity(
                model, latents, t_cur, prompt_embeds, image_latents,
                scale=cfg_scale, uncond_embeds=uncond_embeds,
            )
            dt = (t_next - t_cur) / 1000.0
            if stochastic:
                sigma = (t_cur.float() / 1000.0).reshape(1)
                sigma_next = (t_next.float() / 1000.0).reshape(1)
                next_latents, old_log_prob, _, _ = sde_step(
                    v, sigma, sigma_next, latents,
                    noise_level=float(self.cfg.training.get("flowopd_noise_level", 0.7)),
                )
                trajectory.append(FlowState(
                    latents=latents.detach().clone(), timestep=t_cur.detach().clone(),
                    extra={
                        "next_latents": next_latents.detach().clone(),
                        "timestep_next": t_next.detach().clone(),
                        "sigma": sigma.detach().clone(),
                        "sigma_next": sigma_next.detach().clone(),
                        "old_log_prob": old_log_prob.detach().clone(),
                    },
                ))
                latents = next_latents
            else:
                trajectory.append(FlowState(latents=latents.detach().clone(), timestep=t_cur.detach().clone()))
                latents = latents + dt * v
            latents = latents.to(self.torch_dtype)
        return trajectory, latents.detach()

    def _flowopd_loss(
        self, prompt_embeds, image_latents, teacher, raw_student,
        teacher_cfg_scale: float, student_cfg_scale: float, uncond_embeds=None,
    ) -> torch.Tensor:
        anchor_name = self.cfg.training.get("flowopd_mar_teacher")
        anchor = self.teachers.get(str(anchor_name)) if anchor_name else None
        total = None
        count = 0
        global_group_size = int(self.cfg.training.get("flowopd_group_size", 16))
        group_size = max(1, math.ceil(global_group_size / int(getattr(self.accelerator, "num_processes", 1))))
        for _ in range(group_size):
            trajectory, _ = self._rollout(
                raw_student, prompt_embeds, image_latents, stochastic=True,
                cfg_scale=student_cfg_scale, uncond_embeds=uncond_embeds,
            )
            indices = flowopd_query_indices(
                len(trajectory), self.cfg.training.get("flowopd_k_states"),
                str(self.cfg.training.get("flowopd_query_bias", "uniform")),
            )
            for i in indices:
                state = trajectory[i]
                extra = state.extra or {}
                z = state.latents.to(self.device, dtype=self.torch_dtype)
                z_next = extra["next_latents"].to(self.device, dtype=self.torch_dtype)
                t = state.timestep.to(self.device)
                pred = self._cfg_velocity(
                    self.student, z, t, prompt_embeds, image_latents,
                    scale=student_cfg_scale, uncond_embeds=uncond_embeds,
                )
                _, log_prob, mean_student, diffusion_std = sde_step(
                    pred, extra["sigma"], extra["sigma_next"], z,
                    previous_sample=z_next,
                    noise_level=float(self.cfg.training.get("flowopd_noise_level", 0.7)),
                )
                with torch.no_grad():
                    target = self._cfg_velocity(
                        teacher, z, t, prompt_embeds, image_latents,
                        scale=teacher_cfg_scale, uncond_embeds=uncond_embeds,
                    )
                    _, _, mean_teacher, _ = sde_step(
                        target, extra["sigma"], extra["sigma_next"], z,
                        previous_sample=z_next,
                        noise_level=float(self.cfg.training.get("flowopd_noise_level", 0.7)),
                    )
                    mean_anchor = None
                    if anchor is not None:
                        anchor_v = self._velocity(anchor, z, t, prompt_embeds, image_latents)
                        _, _, mean_anchor, _ = sde_step(
                            anchor_v, extra["sigma"], extra["sigma_next"], z,
                            previous_sample=z_next,
                            noise_level=float(self.cfg.training.get("flowopd_noise_level", 0.7)),
                        )
                loss, _ = clipped_policy_loss(
                    log_prob=log_prob, old_log_prob=extra["old_log_prob"].to(self.device),
                    mean_student=mean_student, mean_teacher=mean_teacher, diffusion_std=diffusion_std,
                    clip_range=float(self.cfg.training.get("flowopd_clip_range", 1e-4)),
                    kl_scale=float(self.cfg.training.get("flowopd_kl_scale", -1.0)),
                    advantage_clip=float(self.cfg.training.get("flowopd_adv_clip_max", 5.0)),
                    mean_anchor=mean_anchor,
                    anchor_beta=float(self.cfg.training.get("flowopd_anchor_beta", 0.0)),
                )
                total = loss if total is None else total + loss
                count += 1
        if total is None:
            raise RuntimeError("FlowOPD produced no query states")
        return total / count

    @staticmethod
    def _load_source_image(path: str):
        from PIL import Image

        return Image.open(path).convert("RGB")

    @torch.no_grad()
    def _encode_source_image(self, image_or_path):
        image = (
            self._load_source_image(image_or_path)
            if isinstance(image_or_path, (str, os.PathLike))
            else image_or_path
        )
        height = int(self.cfg.training.height or self.cfg.training.resolution)
        width = int(self.cfg.training.width or self.cfg.training.resolution)
        image = image.resize((width, height))
        # DiffSynth releases expose either pipeline.encode_image or a VAE
        # encoder. Keep both explicit so API drift fails with a useful message.
        if hasattr(self.pipe, "encode_image"):
            latent = self.pipe.encode_image(image)
        elif getattr(self.pipe, "vae_encoder", None) is not None:
            if hasattr(self.pipe, "load_models_to_device"):
                self.pipe.load_models_to_device(["vae_encoder"])
            latent = self.pipe.vae_encoder(self.pipe.preprocess_image(image))
        else:
            raise RuntimeError("This DiffSynth ZImagePipeline cannot encode source images; update DiffSynth-Studio")
        if isinstance(latent, tuple):
            latent = latent[0]
        return latent.to(self.device, dtype=self.torch_dtype)

    def _renoise(self, x0: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        scheduler = self.pipe.scheduler
        noise = torch.randn_like(x0)
        if hasattr(scheduler, "add_noise"):
            return scheduler.add_noise(x0, noise, timestep.reshape(1)).to(self.torch_dtype)
        t = (timestep.float() / 1000.0).reshape(1, 1, 1, 1).to(self.device)
        return ((1.0 - t) * x0.float() + t * noise.float()).to(self.torch_dtype)

    def compute_loss(self, sample, route) -> torch.Tensor:
        edit_image = self._load_source_image(sample.source_image) if sample.source_image else None
        prompt_embeds = self._encode_prompt(sample.prompt, edit_image=edit_image)
        image_latents = [self._encode_source_image(edit_image)] if edit_image is not None else None
        teacher = self.teachers[route.teacher]
        method = get_method(self.cfg.training.method)
        raw_student = self.accelerator.unwrap_model(self.student)

        teacher_cfg_scale = float(self.cfg.training.get("teacher_cfg_scale", 1.0))
        student_cfg_scale = float(self.cfg.training.get("student_cfg_scale", 1.0))
        if teacher_cfg_scale != 1.0 or student_cfg_scale != 1.0:
            uncond_embeds = self._encode_prompt("", edit_image=edit_image)
        else:
            uncond_embeds = None
        if method.name == "flowopd":
            return self._flowopd_loss(
                prompt_embeds, image_latents, teacher, raw_student,
                teacher_cfg_scale, student_cfg_scale, uncond_embeds,
            )

        if method.state_source == "offline":
            if sample.target_image:
                x0 = self._encode_source_image(sample.target_image)
            else:
                x0 = torch.randn(self._latent_shape(), device=self.device, dtype=self.torch_dtype)
            timesteps = self._timesteps()
            indices = query_indices(method, len(timesteps), int(self.cfg.training.k), str(self.cfg.training.query_bias))
            states = [FlowState(latents=self._renoise(x0, timesteps[i]).detach(), timestep=timesteps[i].detach()) for i in indices]
        else:
            trajectory, _ = self._rollout(
                raw_student, prompt_embeds, image_latents, stochastic=method.stochastic_rollout,
                cfg_scale=student_cfg_scale, uncond_embeds=uncond_embeds,
            )
            indices = query_indices(method, len(trajectory), int(self.cfg.training.k), str(self.cfg.training.query_bias))
            states = [trajectory[i] for i in indices]

        total = None
        for state in states:
            z = state.latents.to(self.device, dtype=self.torch_dtype)
            t = state.timestep.to(self.device)
            with torch.no_grad():
                target = self._cfg_velocity(
                    teacher, z, t, prompt_embeds, image_latents,
                    scale=teacher_cfg_scale, uncond_embeds=uncond_embeds,
                )
            pred = self._cfg_velocity(
                self.student, z, t, prompt_embeds, image_latents,
                scale=student_cfg_scale, uncond_embeds=uncond_embeds,
            )
            weight = 1.0
            if method.objective == "diffusion_kl":
                weight = 0.5 / float(max(1, int(self.cfg.training.rollout_steps)) ** 2)
            divisor = 1 if method.objective == "diffusion_kl" else len(states)
            loss = weight * velocity_mse(pred, target) / divisor
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
        if self.accelerator.is_main_process:
            out = step_dir(self.cfg.training.output_dir, step)
            os.makedirs(out, exist_ok=True)
            raw = self.accelerator.unwrap_model(self.student)
            from peft import get_peft_model_state_dict

            state = {
                key: value.detach().cpu().contiguous()
                for key, value in get_peft_model_state_dict(raw).items()
            }
            save_file(state, os.path.join(out, "adapter_model.safetensors"))
            peft_configs = getattr(raw, "peft_config", {})
            peft_cfg = peft_configs.get("default") if hasattr(peft_configs, "get") else None
            if peft_cfg is None:
                raise RuntimeError("Z-Image student has no default PEFT config to save")
            peft_cfg.save_pretrained(out)
            save_trainer_state(self.optimizer, out, step)
            print(f"[DanceOPD][zimage] saved {out}", flush=True)
        self.accelerator.wait_for_everyone()

    def resume(self, checkpoint_dir: str) -> int:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        adapter_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
        if not os.path.isfile(adapter_path):
            raise ValueError(f"Resume checkpoint has no adapter_model.safetensors: {checkpoint_dir}")
        raw = self.accelerator.unwrap_model(self.student)
        result = set_peft_model_state_dict(raw, load_file(adapter_path), adapter_name="default")
        unexpected = list(getattr(result, "unexpected_keys", []))
        if unexpected:
            raise RuntimeError(f"Resume adapter has unexpected tensors: {unexpected[:5]}")
        step = load_trainer_state(self.optimizer, checkpoint_dir)
        self._log(f"resumed {checkpoint_dir} at step={step}")
        return step

    def _log(self, msg: str) -> None:
        if self.accelerator.is_main_process:
            print(f"[DanceOPD][zimage] {msg}", flush=True)
