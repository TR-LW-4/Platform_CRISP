"""
Cycle-breaking extensions for LeeHsu2007NetworkFlowIP.

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


def add_exact_final_layout(m, V: ModelVars, dims: ModelDims, demand: Dict[Tuple[int, int], Optional[int]]) -> None:
    """
    Eq. (27): ``demand[(s, h)]`` is the required type *value* at slot h of
    stack s in the final layout, or ``None`` for an empty slot. Any slot
    not present in ``demand`` is left unconstrained (falls back to the
    ordering constraint already present from the basic model).
    """
    C = dims.n_types
    T = dims.n_time_points
    type_to_idx = {v: c for c, v in enumerate(dims.type_values)}

    for (s, h), value in demand.items():
        occ_c = type_to_idx[value] if value is not None else None
        for c in range(C):
            target = 1 if c == occ_c else 0
            m.addConstr(V.ci[(T - 1, s, h, c)] == target, name=f"c27_{s}_{h}_{c}")


def add_one_type_per_stack(m, V: ModelVars, dims: ModelDims) -> None:
    """Eq. (28): each stack holds at most one container type in the final layout."""
    import gurobipy as gp

    S = dims.n_stacks
    H = dims.max_tiers
    C = dims.n_types
    T = dims.n_time_points

    for c1 in range(C):
        for s in range(S):
            for h in range(H - 1):
                m.addConstr(
                    V.ci[(T - 1, s, h + 1, c1)]
                    + gp.quicksum(V.ci[(T - 1, s, h, c2)] for c2 in range(C) if c2 != c1)
                    <= 1,
                    name=f"c28_{c1}_{s}_{h}",
                )
