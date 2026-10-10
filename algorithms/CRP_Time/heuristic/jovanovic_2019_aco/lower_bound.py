"""
Crane-time lower bounds LB_f (f2) and LB_v (f2vert) of JovanovicACO_CRPTime
(Sec. 6, eqs. 44–45, p. 85): the crane time still needed for a bay. They set the
quality of a solution and the early abort of an ant.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List

from algorithms.CRP_Time.utils.located import non_located
from core.objectives import ObjectiveSpec


def lb_container(s: int, t: int, nw: bool, spec: ObjectiveSpec) -> float:
    """
    Contribution of one container at stack s, tier t to LB_f (eq. 44) or
    LB_v (eq. 45): its retrieval from the current position, plus the
    minimal relocation cost if it is non-well-located (nw).

    Taken literally from the paper; LB_f is not always a valid lower bound
    (see docs/validation/jovanovic_2019.md §3.6).
    """
    ts = spec.stack_s_per_stack
    if spec.time_model == "f2":
        tpp = spec.pickup_place_s
        return tpp + 2.0 * s * ts + ((tpp + ts) if nw else 0.0)
    tr = spec.empty_vertical_s_per_tier + spec.loaded_vertical_s_per_tier
    h_max = spec.max_tiers + 1
    return (
        (2.0 * h_max - t - spec.outside_height) * tr
        + 2.0 * s * ts
        + ((2.0 * tr + ts) if nw else 0.0)
    )


def lb_time(stacks: List[List[int]], spec: ObjectiveSpec) -> float:
    """LB_f or LB_v of a bay; stack s of the paper is stacks[s - 1], tiers 1-based from the ground."""
    total = 0.0
    for s, stack in enumerate(stacks, 1):
        for t, nw in enumerate(non_located(stack), 1):
            total += lb_container(s, t, nw, spec)
    return total
