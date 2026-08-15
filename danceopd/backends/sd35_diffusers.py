"""Stable Diffusion 3.5 / Diffusers backend for DanceOPD."""
from __future__ import annotations

import math
import os
from typing import Any

import torch

from danceopd.backends.base import DanceOPDBackend
from danceopd.backends.teacher import compose_teacher, load_compatible_state_dict, load_state_dict_file
from danceopd.core.checkpoint import ensure_output_dir, load_trainer_state, save_trainer_state, step_dir
from danceopd.core.config import teacher_map
from danceopd.core.flowopd import clipped_policy_loss, sde_step
from danceopd.core.loss import velocity_mse
from danceopd.core.methods import flowopd_query_indices, get_method, query_indices
from danceopd.core.rollout import FlowState
from danceopd.core.timestep import cfg_target


def _dtype(name: str):
    name = (name or "bf16").lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


class SD35Backend(DanceOPDBackend):
    """DanceOPD backend for SD3.5 rectified-flow transformers.

    Teacher: frozen SD3 transformer, optionally initialized from a full
    transformer checkpoint and optionally merged with a PEFT LoRA.
    Student: base or warm-started SD3 transformer plus a trainable LoRA.
    """

    def __init__(self, cfg, accelerator):
        self.cfg = cfg
        self.accelerator = accelerator
        self.device = accelerator.device
        self.torch_dtype = _dtype(cfg.training.mixed_precision)
        self.pipe = None
        self.student = None
        self.optimizer = None
        self.teachers: dict[str, Any] = {}
        self.scheduler = None
        self.trainable_params = []

    def prepare(self) -> None:
        from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
        from peft import LoraConfig

        model_root = self.cfg.model.pretrained_model
        self.pipe = StableDiffusion3Pipeline.from_pretrained(model_root, torch_dtype=self.torch_dtype)
        self.scheduler = self.pipe.scheduler
        student = self.pipe.transformer

        student_init = self._default_student_init(self.cfg.student.init)
        if student_init and student_init not in {"base", "none"}:
            init_path = self._resolve_student_init(student_init)
            if init_path:
                if os.path.isfile(str(init_path)):
                    sd = load_state_dict_file(init_path)
                    load_compatible_state_dict(
                        student, sd, label="student warm-start", log=self._log, min_model_match=0.5
                    )
                else:
                    from peft import PeftModel
                    student = PeftModel.from_pretrained(student, str(init_path), is_trainable=False).merge_and_unload()
                    self._log(f"student warm-start PEFT LoRA merged from {init_path}")

        student.requires_grad_(False)
        target_modules = self.cfg.student.lora_target_modules or [
            "to_q", "to_k", "to_v", "to_out.0",
            "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
        ]
        lora_rank = int(self.cfg.student.lora_rank)
        lora_alpha = int(self.cfg.student.lora_alpha or lora_rank)
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        student.add_adapter(lora_cfg)
        if hasattr(student, "enable_gradient_checkpointing"):
            student.enable_gradient_checkpointing()

        for text_encoder in (self.pipe.text_encoder, self.pipe.text_encoder_2, self.pipe.text_encoder_3):
            if text_encoder is not None:
                text_encoder.requires_grad_(False).eval().to(self.device)
        if getattr(self.pipe, "vae", None) is not None:
            self.pipe.vae.requires_grad_(False).eval().to("cpu")

        self.teachers = {}
        for name, tcfg in teacher_map(self.cfg).items():
            base_ckpt = tcfg.get("base_ckpt")
            if base_ckpt and os.path.isdir(str(base_ckpt)):
                try:
                    teacher = SD3Transformer2DModel.from_pretrained(str(base_ckpt), torch_dtype=self.torch_dtype)
                except (OSError, ValueError):
                    teacher = SD3Transformer2DModel.from_pretrained(
                        str(base_ckpt), subfolder="transformer", torch_dtype=self.torch_dtype
                    )
                self._log(f"teacher={name} full checkpoint directory loaded")
                teacher = compose_teacher(
                    teacher,
                    {**tcfg, "base_ckpt": None},
                    label=f"teacher={name}",
                    log=self._log,
                    device=self.device,
                )
            else:
                teacher = SD3Transformer2DModel.from_pretrained(
                    model_root, subfolder="transformer", torch_dtype=self.torch_dtype
                )
                teacher = compose_teacher(teacher, tcfg, label=f"teacher={name}", log=self._log, device=self.device)
            self.teachers[name] = teacher

        self.trainable_params = [p for p in student.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            self.trainable_params,
            lr=float(self.cfg.training.lr),
            weight_decay=float(self.cfg.training.weight_decay),
        )
        self.student, self.optimizer = self.accelerator.prepare(student, self.optimizer)
        ensure_output_dir(self.cfg.training.output_dir)
        n_train = sum(p.numel() for p in self.trainable_params)
        self._log(f"student LoRA rank={lora_rank} trainable_params={n_train/1e6:.2f}M")

    def _resolve_student_init(self, value: str) -> str | None:
        if value in {"teacher", "teacher_base"}:
            teachers = list(teacher_map(self.cfg).values())
            return teachers[0].get("base_ckpt") if teachers else None
        return value

    def _default_student_init(self, value: str | None) -> str:
        if value in {None, "", "auto"}:
            # Public, directly downloadable OCR expert used by Flow-OPD.
            return "jieliu/SD3.5M-FlowGRPO-Text"
        return str(value)

    @torch.no_grad()
    def _encode_prompt(self, prompt: str):
        prompt_embeds, _, pooled, _ = self.pipe.encode_prompt(
            prompt=[prompt], prompt_2=[prompt], prompt_3=[prompt],
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            max_sequence_length=int(self.cfg.training.max_sequence_length),
        )
        return prompt_embeds.to(self.torch_dtype), pooled.to(self.torch_dtype)

    def _latent_shape(self) -> tuple[int, int, int, int]:
        raw_student = self.accelerator.unwrap_model(self.student) if self.student is not None else self.pipe.transformer
        latent_ch = int(raw_student.config.in_channels)
        height = int(self.cfg.training.height or self.cfg.training.resolution)
        width = int(self.cfg.training.width or self.cfg.training.resolution)
        return (1, latent_ch, height // 8, width // 8)

    def _model_velocity(self, transformer, latents: torch.Tensor, timestep: torch.Tensor, prompt_embeds, pooled):
        latents = latents.to(self.torch_dtype)
        bsz = latents.shape[0]
        ts = timestep.reshape(1).expand(bsz).to(device=latents.device, dtype=self.torch_dtype)
        return transformer(
            hidden_states=latents,
            timestep=ts,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled,
            return_dict=False,
        )[0]

    def _cfg_velocity(
        self, transformer, latents, timestep, prompt_embeds, pooled,
        *, scale: float = 1.0, uncond_embeds=None, uncond_pooled=None,
    ):
        cond = self._model_velocity(transformer, latents, timestep, prompt_embeds, pooled)
        if float(scale) == 1.0:
            return cond
        if uncond_embeds is None or uncond_pooled is None:
            raise ValueError("CFG scale != 1 requires unconditional prompt embeddings")
        uncond = self._model_velocity(transformer, latents, timestep, uncond_embeds, uncond_pooled)
        return cfg_target(uncond, cond, float(scale))

    @torch.no_grad()
    def _rollout(
        self, model, prompt_embeds, pooled, *, stochastic: bool = False,
        cfg_scale: float = 1.0, uncond_embeds=None, uncond_pooled=None,
    ) -> tuple[list[FlowState], torch.Tensor]:
        steps = int(self.cfg.training.rollout_steps)
        self.scheduler.set_timesteps(steps, device=self.device)
        sigmas = self.scheduler.sigmas.to(self.device)
        timesteps = self.scheduler.timesteps.to(self.device)
        latents = torch.randn(self._latent_shape(), device=self.device, dtype=self.torch_dtype)
        trajectory: list[FlowState] = []
        for i in range(steps):
            t = timesteps[i]
            v = self._cfg_velocity(
                model, latents, t, prompt_embeds, pooled, scale=cfg_scale,
                uncond_embeds=uncond_embeds, uncond_pooled=uncond_pooled,
            )
            dt = sigmas[i + 1] - sigmas[i]
            if stochastic:
                next_latents, old_log_prob, _, _ = sde_step(
                    v, sigmas[i:i + 1], sigmas[i + 1:i + 2], latents,
                    noise_level=float(self.cfg.training.get("flowopd_noise_level", 0.7)),
                )
                trajectory.append(FlowState(
                    latents=latents.detach().clone(), timestep=t.detach().clone(),
                    extra={
                        "next_latents": next_latents.detach().clone(),
                        "sigma": sigmas[i].detach().clone(),
                        "sigma_next": sigmas[i + 1].detach().clone(),
                        "old_log_prob": old_log_prob.detach().clone(),
                    },
                ))
                latents = next_latents
            else:
                trajectory.append(FlowState(latents=latents.detach().clone(), timestep=t.detach().clone()))
                latents = latents + dt * v
            latents = latents.to(self.torch_dtype)
        return trajectory, latents.detach()

    def _flowopd_loss(
        self, prompt_embeds, pooled, teacher, raw_student,
        teacher_cfg_scale: float, student_cfg_scale: float,
    ) -> torch.Tensor:
        if teacher_cfg_scale != 1.0 or student_cfg_scale != 1.0:
            uncond_embeds, uncond_pooled = self._encode_prompt("")
        else:
            uncond_embeds = uncond_pooled = None
        anchor_name = self.cfg.training.get("flowopd_mar_teacher")
        anchor = self.teachers.get(str(anchor_name)) if anchor_name else None
        total = None
        count = 0
        global_group_size = int(self.cfg.training.get("flowopd_group_size", 16))
        group_size = max(1, math.ceil(global_group_size / int(getattr(self.accelerator, "num_processes", 1))))
        for _ in range(group_size):
            trajectory, _ = self._rollout(
                raw_student, prompt_embeds, pooled, stochastic=True,
                cfg_scale=student_cfg_scale, uncond_embeds=uncond_embeds,
                uncond_pooled=uncond_pooled,
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
                    self.student, z, t, prompt_embeds, pooled, scale=student_cfg_scale,
                    uncond_embeds=uncond_embeds, uncond_pooled=uncond_pooled,
                )
                _, log_prob, mean_student, diffusion_std = sde_step(
                    pred, extra["sigma"].reshape(1), extra["sigma_next"].reshape(1), z,
                    previous_sample=z_next,
                    noise_level=float(self.cfg.training.get("flowopd_noise_level", 0.7)),
                )
                with torch.no_grad():
                    target = self._cfg_velocity(
                        teacher, z, t, prompt_embeds, pooled, scale=teacher_cfg_scale,
                        uncond_embeds=uncond_embeds, uncond_pooled=uncond_pooled,
                    )
                    _, _, mean_teacher, _ = sde_step(
                        target, extra["sigma"].reshape(1), extra["sigma_next"].reshape(1), z,
                        previous_sample=z_next,
                        noise_level=float(self.cfg.training.get("flowopd_noise_level", 0.7)),
                    )
                    mean_anchor = None
                    if anchor is not None:
                        anchor_v = self._model_velocity(anchor, z, t, prompt_embeds, pooled)
                        _, _, mean_anchor, _ = sde_step(
                            anchor_v, extra["sigma"].reshape(1), extra["sigma_next"].reshape(1), z,
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

    def _sigma_for_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        timesteps = self.scheduler.timesteps.to(self.device)
        sigmas = self.scheduler.sigmas.to(self.device)
        idx = (timesteps == timestep).nonzero(as_tuple=False)
        if idx.numel() == 0:
            nearest = torch.argmin(torch.abs(timesteps.float() - timestep.float()))
            return sigmas[nearest]
        return sigmas[idx.flatten()[0]]

    def compute_loss(self, sample, route) -> torch.Tensor:
        if sample.source_image:
            raise ValueError("The SD3.5-M public backend is T2I-only; use backend=zimage for source-image edit rows")
        prompt_embeds, pooled = self._encode_prompt(sample.prompt)
        teacher = self.teachers[route.teacher]
        method = get_method(self.cfg.training.method)
        raw_student = self.accelerator.unwrap_model(self.student)

        teacher_cfg_scale = float(self.cfg.training.get("teacher_cfg_scale", 1.0))
        student_cfg_scale = float(self.cfg.training.get("student_cfg_scale", 1.0))
        if method.name == "flowopd":
            return self._flowopd_loss(
                prompt_embeds, pooled, teacher, raw_student,
                teacher_cfg_scale, student_cfg_scale,
            )
        if teacher_cfg_scale != 1.0 or student_cfg_scale != 1.0:
            uncond_embeds, uncond_pooled = self._encode_prompt("")
        else:
            uncond_embeds = uncond_pooled = None

        if method.state_source == "offline":
            # Paper DiffusionOPD comparison: independent forward-noised offline
            # endpoint. T2I uses a random latent endpoint; edit rows require an
            # encoded target and are handled by the Z-Image backend.
            x0 = torch.randn(self._latent_shape(), device=self.device, dtype=self.torch_dtype)
            self.scheduler.set_timesteps(int(self.cfg.training.rollout_steps), device=self.device)
            indices = query_indices(method, int(self.cfg.training.rollout_steps), int(self.cfg.training.k), str(self.cfg.training.query_bias))
            states: list[FlowState] = []
            for i in indices:
                t = self.scheduler.timesteps.to(self.device)[i]
                sigma = self._sigma_for_timestep(t).to(dtype=self.torch_dtype)
                noise = torch.randn_like(x0)
                z = ((1.0 - sigma) * x0 + sigma * noise).to(self.torch_dtype)
                states.append(FlowState(latents=z.detach(), timestep=t.detach()))
        else:
            trajectory, _ = self._rollout(
                raw_student, prompt_embeds, pooled, stochastic=method.stochastic_rollout,
                cfg_scale=student_cfg_scale, uncond_embeds=uncond_embeds,
                uncond_pooled=uncond_pooled,
            )
            indices = query_indices(method, len(trajectory), int(self.cfg.training.k), str(self.cfg.training.query_bias))
            states = [trajectory[i] for i in indices]

        total = None
        for state in states:
            z = state.latents.to(self.device, dtype=self.torch_dtype)
            t = state.timestep.to(self.device)
            with torch.no_grad():
                target = self._cfg_velocity(
                    teacher, z, t, prompt_embeds, pooled, scale=teacher_cfg_scale,
                    uncond_embeds=uncond_embeds, uncond_pooled=uncond_pooled,
                )
            pred = self._cfg_velocity(
                self.student, z, t, prompt_embeds, pooled, scale=student_cfg_scale,
                uncond_embeds=uncond_embeds, uncond_pooled=uncond_pooled,
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
            raw_student = self.accelerator.unwrap_model(self.student)
            # Export only the standard PEFT adapter, not the full transformer.
            from peft import get_peft_model_state_dict
            from safetensors.torch import save_file

            state = {
                key: value.detach().cpu().contiguous()
                for key, value in get_peft_model_state_dict(raw_student).items()
            }
            save_file(state, os.path.join(out, "adapter_model.safetensors"))
            peft_configs = getattr(raw_student, "peft_config", {})
            peft_cfg = peft_configs.get("default") if hasattr(peft_configs, "get") else None
            if peft_cfg is None:
                raise RuntimeError("SD3 student has no default PEFT config to save")
            peft_cfg.save_pretrained(out)
            save_trainer_state(self.optimizer, out, step)
            print(f"[DanceOPD][sd35] saved {out}", flush=True)
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
            print(f"[DanceOPD][sd35] {msg}", flush=True)
