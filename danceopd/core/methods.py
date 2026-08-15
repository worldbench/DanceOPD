"""Training-method contracts shared by all model backends.

The three names intentionally describe different state distributions/objectives;
they are not aliases for the same velocity-MSE loop.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSpec:
    name: str
    state_source: str
    stochastic_rollout: bool
    dense: bool
    objective: str


METHODS = {
    # DanceOPD: student ODE trajectory, normally one Beta(5,2)-biased query.
    "danceopd": MethodSpec("danceopd", "student", False, False, "velocity_mse"),
    # DiffusionOPD: dense ODE trajectory KL (Eq. 12); under a uniform grid this
    # is sum_k (dt^2/2) * ||v_student-v_teacher||^2.
    "diffusionopd": MethodSpec("diffusionopd", "student", False, True, "diffusion_kl"),
    # Flow-OPD: stochastic student trajectory and transition-distribution KL.
    "flowopd": MethodSpec("flowopd", "student", True, True, "transition_kl"),
    # Paper ablation, not one of the three primary methods: offline endpoint
    # forward-noising without any student or teacher rollout.
    "offpolicy": MethodSpec("offpolicy", "offline", False, False, "velocity_mse"),
}


def get_method(name: str) -> MethodSpec:
    key = str(name or "danceopd").lower().replace("-", "")
    aliases = {
        "diffusionopd": "diffusionopd", "flowopd": "flowopd",
        "danceopd": "danceopd", "offpolicy": "offpolicy",
    }
    try:
        return METHODS[aliases[key]]
    except KeyError as exc:
        raise ValueError(f"Unknown method {name!r}; choose from {sorted(METHODS)}") from exc


def query_indices(spec: MethodSpec, n_states: int, configured_k: int, bias: str) -> list[int]:
    from danceopd.core.timestep import sample_query_indices

    k = n_states if spec.dense else configured_k
    return sample_query_indices(n_states, k, bias)


def flowopd_query_indices(n_states: int, configured_k: int | None, bias: str = "uniform") -> list[int]:
    """Flow-OPD's K-state selector (dense when K is unset)."""
    from danceopd.core.timestep import sample_query_indices

    k = n_states if configured_k in (None, 0) else int(configured_k)
    return sample_query_indices(n_states, k, bias)
