"""Teacher routing utilities."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Route:
    teacher: str
    weight: float = 1.0


class WeightedRouter:
    def __init__(self, routes: Iterable[Route]):
        self.routes = list(routes)
        if not self.routes:
            raise ValueError("At least one route is required.")
        if any(r.weight < 0 for r in self.routes):
            raise ValueError("Route weights must be non-negative.")
        if sum(r.weight for r in self.routes) <= 0:
            raise ValueError("At least one route must have positive weight.")

    @classmethod
    def from_config(cls, cfg) -> "WeightedRouter":
        routes = []
        for item in cfg.get_dotted("routing.routes", []):
            routes.append(Route(teacher=str(item.get("teacher", "default")), weight=float(item.get("weight", 1.0))))
        return cls(routes)

    def sample(self) -> Route:
        return random.choices(self.routes, weights=[r.weight for r in self.routes], k=1)[0]
