"""
LIFO/flow strengthening for DeMeloSilva2018PMP.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from .mip_model import ModelDims, ModelVars


def add_exact_final_layout(
    m,
    V: ModelVars,
    dims: ModelDims,
    demand: Dict[Tuple[int, int], Optional[int]],
) -> None:
    """
    Eq. (6): ``demand[(s, h)]`` is the required group *value* at slot h of
    stack s in the final layout (time point R), or ``None`` for an empty
    slot. Slots not present in ``demand`` are left unconstrained.
    """
    G = dims.n_groups
    R = dims.n_steps
    group_to_idx = {v: g for g, v in enumerate(dims.group_values)}

    for (s, h), value in demand.items():
        occ_g = group_to_idx[value] if value is not None else None
        for g in range(G):
            target = 1 if g == occ_g else 0
            m.addConstr(V.x[(R, g, s, h)] == target, name=f"c6_{s}_{h}_{g}")


def add_sorted_final_layout(m, V: ModelVars, dims: ModelDims) -> None:
    """Eq. (7): in the final layout, the group value directly below slot h
    must be >= the group value at slot h (i.e. less-urgent / larger-valued
    groups sink to the bottom of each stack)."""
    import gurobipy as gp

    S = dims.n_stacks
    H = dims.max_tiers
    G = dims.n_groups
    R = dims.n_steps

    for g in range(G):
        for s in range(S):
            for h in range(1, H):
                m.addConstr(
                    gp.quicksum(V.x[(R, q, s, h - 1)] for q in range(g, G)) >= V.x[(R, g, s, h)],
                    name=f"c7_{g}_{s}_{h}",
                )
