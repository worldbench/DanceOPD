"""Query-state sampling for DanceOPD."""
from __future__ import annotations

import random


def sample_query_indices(n_states: int, k: int = 1, bias: str = "low_t") -> list[int]:
    """Sample trajectory indices.

    Trajectory order is high-noise to low-noise. `low_t` therefore means indices
    near the end of the rollout, using the default DanceOPD Beta(5, 2) bias.
    """
    if n_states <= 0:
        raise ValueError("n_states must be positive")
    k = max(1, min(int(k), n_states))
    bias = (bias or "low_t").lower()

    if k >= n_states:
        return list(range(n_states))

    if bias == "uniform":
        return sorted(random.sample(range(n_states), k))

    if bias == "stratified":
        width = n_states / k
        return sorted(min(int((i + random.random()) * width), n_states - 1) for i in range(k))

    if bias == "high_t":
        alpha, beta = 2.0, 5.0
    elif bias == "mid_t":
        alpha, beta = 5.0, 5.0
    else:
        alpha, beta = 5.0, 2.0

    chosen: set[int] = set()
    attempts = 0
    while len(chosen) < k and attempts < 50 * k:
        u = random.betavariate(alpha, beta)
        chosen.add(min(max(int(u * n_states), 0), n_states - 1))
        attempts += 1
    while len(chosen) < k:
        chosen.add(random.randrange(n_states))
    return sorted(chosen)
