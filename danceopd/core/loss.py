"""Loss functions."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def velocity_mse(student_velocity: torch.Tensor, teacher_velocity: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(student_velocity.float(), teacher_velocity.detach().float())
