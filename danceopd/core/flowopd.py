"""Flow-OPD on-policy SDE transition and PPO objective.

This is the backend-independent part of the updated Table-2 implementation.
Backends provide their native sigma schedule and velocity networks; this module
keeps the transition distribution and clipped policy objective identical across
SD3.5-M and Z-Image.
"""
from __future__ import annotations

import math

import torch


def _expand(value: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
    value = value.to(device=sample.device, dtype=torch.float32).reshape(-1)
    return value.reshape(value.shape[0], *([1] * (sample.ndim - 1)))


def sde_step(
    model_output: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    sample: torch.Tensor,
    *,
    previous_sample: torch.Tensor | None = None,
    noise_level: float = 0.7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample/evaluate one Flow-OPD SDE transition.

    Returns ``(next_sample, log_prob, transition_mean, diffusion_std)``.  The
    formula follows Flow-OPD's ``sd3_sde_with_logprob.py`` and the corresponding
    Z-Image implementation. ``sigma_next < sigma`` is required.
    """
    x = sample.float()
    v = model_output.float()
    sigma = _expand(sigma, x).clamp(1e-4, 1.0)
    sigma_next = _expand(sigma_next, x).clamp(0.0, 1.0)
    sigma_safe = torch.where(sigma >= 0.9999, sigma_next.clamp(min=1e-4, max=0.999), sigma)
    dt = sigma_next - sigma
    if torch.any(dt >= 0):
        raise ValueError("Flow-OPD requires a strictly decreasing sigma schedule")

    diffusion_std = torch.sqrt(sigma / (1.0 - sigma_safe).clamp_min(1e-4)) * float(noise_level)
    transition_mean = x * (1.0 + diffusion_std.square() / (2.0 * sigma) * dt) + v * (
        1.0 + diffusion_std.square() * (1.0 - sigma) / (2.0 * sigma)
    ) * dt
    step_std = diffusion_std * torch.sqrt((-dt).clamp_min(1e-6))
    if previous_sample is None:
        previous_sample = transition_mean + step_std * torch.randn_like(transition_mean)
    else:
        previous_sample = previous_sample.float()

    log_prob = -((previous_sample.detach() - transition_mean).square()) / (
        2.0 * step_std.square().clamp_min(1e-12)
    ) - torch.log(step_std.clamp_min(1e-12)) - 0.5 * math.log(2.0 * math.pi)
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    return previous_sample.to(sample.dtype), log_prob, transition_mean, diffusion_std


def clipped_policy_loss(
    *,
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    mean_student: torch.Tensor,
    mean_teacher: torch.Tensor,
    diffusion_std: torch.Tensor,
    clip_range: float = 1e-4,
    kl_scale: float = -1.0,
    advantage_clip: float = 5.0,
    mean_anchor: torch.Tensor | None = None,
    anchor_beta: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Flow-OPD KL reward, PPO clipping, and optional MAR anchor KL."""
    reduce_dims = tuple(range(1, mean_student.ndim))
    teacher_kl = (mean_student.detach() - mean_teacher.detach()).square().mean(
        dim=reduce_dims, keepdim=True
    ) / (2.0 * diffusion_std.detach().square().clamp_min(1e-8))
    advantage = (float(kl_scale) * teacher_kl.reshape(teacher_kl.shape[0], -1).mean(-1)).clamp(
        -float(advantage_clip), float(advantage_clip)
    )
    ratio = torch.exp(log_prob - old_log_prob.detach())
    unclipped = -advantage * ratio
    clipped = -advantage * torch.clamp(ratio, 1.0 - float(clip_range), 1.0 + float(clip_range))
    loss = torch.maximum(unclipped, clipped).mean()

    anchor_kl = torch.zeros((), device=loss.device, dtype=loss.dtype)
    if float(anchor_beta) > 0.0:
        if mean_anchor is None:
            raise ValueError("flowopd_anchor_beta > 0 requires flowopd_mar_teacher")
        anchor_kl = (mean_student - mean_anchor.detach()).square().mean(
            dim=reduce_dims, keepdim=True
        ) / (2.0 * diffusion_std.square().clamp_min(1e-8))
        anchor_kl = anchor_kl.mean()
        loss = loss + float(anchor_beta) * anchor_kl

    diagnostics = {
        "teacher_kl": teacher_kl.mean().detach(),
        "anchor_kl": anchor_kl.detach(),
        "clip_fraction": (torch.abs(ratio.detach() - 1.0) > float(clip_range)).float().mean(),
    }
    return loss, diagnostics
