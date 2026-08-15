"""Small local backend used only for zero-weight smoke tests.

The toy backend exercises the DanceOPD engine, routing, prompt loader,
query-state sampling, optimizer step, and checkpoint save path without loading
any diffusion model weights. It is intentionally not a research baseline.
"""
from __future__ import annotations

import hashlib
import os

import torch
from safetensors.torch import save_file

from danceopd.backends.base import DanceOPDBackend
from danceopd.core.checkpoint import ensure_output_dir, load_trainer_state, save_trainer_state, step_dir
from danceopd.core.loss import velocity_mse
from danceopd.core.methods import get_method, query_indices


class ToyBackend(DanceOPDBackend):
    def __init__(self, cfg, accelerator):
        self.cfg = cfg
        self.accelerator = accelerator
        self.device = accelerator.device
        self.student = torch.nn.Linear(8, 8, bias=False)
        self.optimizer = None
        self.trainable_params = list(self.student.parameters())

    def prepare(self) -> None:
        with torch.no_grad():
            self.student.weight.copy_(torch.eye(8) * 0.5)
        self.optimizer = torch.optim.AdamW(
            self.trainable_params,
            lr=float(self.cfg.training.lr),
            weight_decay=float(self.cfg.training.weight_decay),
        )
        self.student, self.optimizer = self.accelerator.prepare(self.student, self.optimizer)
        ensure_output_dir(self.cfg.training.output_dir)
        if self.accelerator.is_main_process:
            print("[DanceOPD][toy] zero-weight smoke backend ready", flush=True)

    def _prompt_vector(self, prompt: str) -> torch.Tensor:
        digest = hashlib.sha256(prompt.encode("utf-8")).digest()
        values = [(digest[i] / 255.0) * 2.0 - 1.0 for i in range(8)]
        return torch.tensor(values, device=self.device, dtype=torch.float32).reshape(1, 8)

    def _teacher_velocity(self, x: torch.Tensor, route, query_index: int) -> torch.Tensor:
        route_scale = 1.0 + (abs(hash(route.teacher)) % 7) / 20.0
        time_scale = 1.0 + query_index / max(1, int(self.cfg.training.rollout_steps))
        return x.roll(shifts=1, dims=-1) * route_scale + 0.1 * time_scale

    def compute_loss(self, sample, route) -> torch.Tensor:
        x = self._prompt_vector(sample.prompt)
        method = get_method(self.cfg.training.method)
        indices = query_indices(
            method, int(self.cfg.training.rollout_steps), int(self.cfg.training.k), str(self.cfg.training.query_bias)
        )
        total = None
        for idx in indices:
            state = x + float(idx) / max(1, int(self.cfg.training.rollout_steps))
            pred = self.student(state)
            target = self._teacher_velocity(state, route, idx)
            weight = 1.0
            if method.objective == "diffusion_kl":
                weight = 0.5 / float(max(1, int(self.cfg.training.rollout_steps)) ** 2)
            elif method.objective == "transition_kl":
                weight = 1.0 / (2.0 * float(self.cfg.training.get("flowopd_noise_level", 0.7)) ** 2)
            divisor = 1 if method.objective == "diffusion_kl" else len(indices)
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
            state = {k: v.detach().cpu().contiguous() for k, v in raw.state_dict().items()}
            save_file(state, os.path.join(out, "toy_student.safetensors"))
            save_trainer_state(self.optimizer, out, step)
            print(f"[DanceOPD][toy] saved {out}", flush=True)
        self.accelerator.wait_for_everyone()

    def resume(self, checkpoint_dir: str) -> int:
        from safetensors.torch import load_file

        raw = self.accelerator.unwrap_model(self.student)
        raw.load_state_dict(load_file(os.path.join(checkpoint_dir, "toy_student.safetensors")))
        step = load_trainer_state(self.optimizer, checkpoint_dir)
        if self.accelerator.is_main_process:
            print(f"[DanceOPD][toy] resumed {checkpoint_dir} at step={step}", flush=True)
        return step
