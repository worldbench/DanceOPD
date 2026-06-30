"""Backend-agnostic rollout data structures."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class FlowState:
    latents: torch.Tensor
    timestep: torch.Tensor
    extra: dict[str, Any] | None = None
