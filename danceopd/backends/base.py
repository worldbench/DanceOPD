"""Backend protocol."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class DanceOPDBackend(ABC):
    @abstractmethod
    def prepare(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, prompt: str, route) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def backward(self, loss: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def optimizer_step(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self, step: int) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return None
