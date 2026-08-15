"""Teacher routing utilities."""
from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Route:
    teacher: str
    weight: float = 1.0
    dataset: str = "default"
    requires_source_image: bool = False


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
    def from_config(cls, cfg) -> WeightedRouter:
        routes = []
        for item in cfg.get_dotted("routing.routes", []):
            routes.append(Route(
                teacher=str(item.get("teacher", "default")),
                weight=float(item.get("weight", 1.0)),
                dataset=str(item.get("dataset", item.get("task", "default"))),
                requires_source_image=bool(item.get("requires_source_image", False)),
            ))
        return cls(routes)

    def sample(self) -> Route:
        return random.choices(self.routes, weights=[r.weight for r in self.routes], k=1)[0]

    def accumulation_routes(self, groups) -> list[Route]:
        """Resolve G=M capability groups without breaking data/teacher binding."""
        if not groups:
            return [self.sample()]
        selected = []
        for group in groups:
            key = str(group)
            matches = [r for r in self.routes if r.dataset == key]
            if not matches:
                matches = [r for r in self.routes if r.teacher == key]
            if len(matches) != 1:
                raise ValueError(f"Accumulation group {key!r} must identify exactly one route, got {len(matches)}")
            selected.append(matches[0])
        return selected
