"""Stable Diffusion 3.5 / Diffusers backend for DanceOPD."""
from __future__ import annotations

import os
from typing import Any

import torch

from danceopd.backends.base import DanceOPDBackend
from danceopd.backends.teacher import compose_teacher, load_state_dict_file
from danceopd.core.checkpoint import ensure_output_dir, step_dir
from danceopd.core.config import teacher_map
from danceopd.core.loss import velocity_mse
from danceopd.core.rollout import FlowState
from danceopd.core.timestep import sample_query_indices


NEGATIVE_PROMPT = ""


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

        student_init = self.cfg.student.init
        if student_init and student_init not in {"base", "none"}:
            init_path = self._resolve_student_init(student_init)
            if init_path:
                sd = load_state_dict_file(init_path)
                missing, unexpected = student.load_state_dict(sd, strict=False)
                self._log(f"student warm-start loaded tensors={len(sd)} missing={len(missing)} unexpected={len(unexpected)}")

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

    @torch.no_grad()
    def _rollout(self, model, prompt_embeds, pooled) -> tuple[list[FlowState], torch.Tensor]:
        steps = int(self.cfg.training.rollout_steps)
        self.scheduler.set_timesteps(steps, device=self.device)
        sigmas = self.scheduler.sigmas.to(self.device)
        timesteps = self.scheduler.timesteps.to(self.device)
        latents = torch.randn(self._latent_shape(), device=self.device, dtype=self.torch_dtype)
        trajectory: list[FlowState] = []
        for i in range(steps):
            t = timesteps[i]
            v = self._model_velocity(model, latents, t, prompt_embeds, pooled)
            trajectory.append(FlowState(latents=latents.detach().clone(), timestep=t.detach().clone()))
            latents = (latents + (sigmas[i + 1] - sigmas[i]) * v).to(self.torch_dtype)
        return trajectory, latents.detach()

    def _sigma_for_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        timesteps = self.scheduler.timesteps.to(self.device)
        sigmas = self.scheduler.sigmas.to(self.device)
        idx = (timesteps == timestep).nonzero(as_tuple=False)
        if idx.numel() == 0:
            nearest = torch.argmin(torch.abs(timesteps.float() - timestep.float()))
            return sigmas[nearest]
        return sigmas[idx.flatten()[0]]

    def compute_loss(self, prompt: str, route) -> torch.Tensor:
        prompt_embeds, pooled = self._encode_prompt(prompt)
        teacher = self.teachers[route.teacher]
        method = str(self.cfg.training.method).lower()
        raw_student = self.accelerator.unwrap_model(self.student)

        if method == "offpolicy":
            _, x0 = self._rollout(teacher, prompt_embeds, pooled)
            self.scheduler.set_timesteps(int(self.cfg.training.rollout_steps), device=self.device)
            indices = sample_query_indices(
                int(self.cfg.training.rollout_steps),
                int(self.cfg.training.k),
                str(self.cfg.training.query_bias),
            )
            states: list[FlowState] = []
            for i in indices:
                t = self.scheduler.timesteps.to(self.device)[i]
                sigma = self._sigma_for_timestep(t).to(dtype=self.torch_dtype)
                noise = torch.randn_like(x0)
                z = ((1.0 - sigma) * x0 + sigma * noise).to(self.torch_dtype)
                states.append(FlowState(latents=z.detach(), timestep=t.detach()))
        else:
            trajectory, _ = self._rollout(raw_student, prompt_embeds, pooled)
            indices = sample_query_indices(len(trajectory), int(self.cfg.training.k), str(self.cfg.training.query_bias))
            states = [trajectory[i] for i in indices]

        total = None
        for state in states:
            z = state.latents.to(self.device, dtype=self.torch_dtype)
            t = state.timestep.to(self.device)
            with torch.no_grad():
                target = self._model_velocity(teacher, z, t, prompt_embeds, pooled)
            pred = self._model_velocity(self.student, z, t, prompt_embeds, pooled)
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
        raw_student = self.accelerator.unwrap_model(self.student)
        raw_student.save_pretrained(out)
        print(f"[DanceOPD][sd35] saved {out}", flush=True)

    def _log(self, msg: str) -> None:
        if self.accelerator.is_main_process:
            print(f"[DanceOPD][sd35] {msg}", flush=True)
