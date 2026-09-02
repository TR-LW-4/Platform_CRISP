"""
PMPm1 Gurobi formulation for DeMeloSilva2018PMP.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .core import Stacks

Key4 = Tuple[int, int, int, int]  # (t, g, s, h)


@dataclass
class ModelDims:
    n_stacks: int
    max_tiers: int
    group_values: List[int]  # group index g (0-based) -> actual priority value
    n_steps: int             # R >= 0 (paper's T): number of relocation steps

    @property
    def n_groups(self) -> int:
        return len(self.group_values)


@dataclass
class ModelVars:
    x: Dict[Key4, object] = field(default_factory=dict)
    y: Dict[Key4, object] = field(default_factory=dict)
    z: Dict[Key4, object] = field(default_factory=dict)


def build_pmp_model(
    stacks_init: Stacks,
    dims: ModelDims,
    time_limit_s: float,
    lifo_strengthening: bool = True,
    flow_strengthening: bool = True,
):
    """
    Builds the Gurobi model for PMPm1. Returns (model, vars, stack_order)
    where ``stack_order`` maps the model's 0-based stack index back to the
    caller's stack keys.
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        raise RuntimeError("Gurobi is required for the de Melo da Silva et al. (2018) PMPm1 model.") from e

    S = dims.n_stacks
    H = dims.max_tiers
    G = dims.n_groups
    R = dims.n_steps
    stack_order = sorted(stacks_init.keys())
    group_to_idx = {v: g for g, v in enumerate(dims.group_values)}

    m = gp.Model("de_melo_silva_2018_pmp_m1")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = max(0.1, float(time_limit_s))

    V = ModelVars()

    for t in range(R + 1):
        for g in range(G):
            for s in range(S):
                for h in range(H):
                    V.x[(t, g, s, h)] = m.addVar(vtype=GRB.BINARY, name=f"x_{t}_{g}_{s}_{h}")

    for t in range(1, R + 1):
        for g in range(G):
            for s in range(S):
                for h in range(H):
                    V.y[(t, g, s, h)] = m.addVar(vtype=GRB.BINARY, name=f"y_{t}_{g}_{s}_{h}")
                    V.z[(t, g, s, h)] = m.addVar(vtype=GRB.BINARY, name=f"z_{t}_{g}_{s}_{h}")

    m.update()

    # C[g,s,h]: fixed constant from the initial layout.
    supply = {}
    for si, s in enumerate(stack_order):
        arr = stacks_init[s]
        for h in range(H):
            occ_g = group_to_idx[arr[h]] if h < len(arr) else None
            for g in range(G):
                supply[(g, si, h)] = 1 if g == occ_g else 0

    # --- Constraint (2): initial layout ----------------------------------
    for g in range(G):
        for s in range(S):
            for h in range(H):
                m.addConstr(V.x[(0, g, s, h)] == supply[(g, s, h)], name=f"c2_{g}_{s}_{h}")

    # --- Constraint (19) [replaces (3)]: at most one group per slot ------
    for t in range(1, R + 1):
        for s in range(S):
            for h in range(H):
                m.addConstr(
                    gp.quicksum(V.x[(t, g, s, h)] + V.z[(t, g, s, h)] for g in range(G)) <= 1,
                    name=f"c19_{t}_{s}_{h}",
                )
    # --- Constraint (5): container-count conservation per group ----------
    for t in range(1, R + 1):
        for g in range(G):
            m.addConstr(
                gp.quicksum(V.x[(t - 1, g, s, h)] for s in range(S) for h in range(H))
                == gp.quicksum(V.x[(t, g, s, h)] for s in range(S) for h in range(H)),
                name=f"c5_{t}_{g}",
            )

    # --- Constraints (8)/(9): at most one relocation per step (K=1) ------
    for t in range(1, R + 1):
        m.addConstr(
            gp.quicksum(V.y[(t, g, s, h)] for g in range(G) for s in range(S) for h in range(H)) <= 1,
            name=f"c8_{t}",
        )
        m.addConstr(
            gp.quicksum(V.z[(t, g, s, h)] for g in range(G) for s in range(S) for h in range(H)) <= 1,
            name=f"c9_{t}",
        )

    # --- Constraint (10): flow link ---------------------------------------
    for t in range(1, R + 1):
        for g in range(G):
            for s in range(S):
                for h in range(H):
                    m.addConstr(
                        V.x[(t, g, s, h)] + V.z[(t, g, s, h)] == V.x[(t - 1, g, s, h)] + V.y[(t, g, s, h)],
                        name=f"c10_{t}_{g}_{s}_{h}",
                    )

    # --- Constraint (11): idle steps pushed to the end --------------------
    for t in range(2, R + 1):
        m.addConstr(
            gp.quicksum(V.z[(t - 1, g, s, h)] for g in range(G) for s in range(S) for h in range(H))
            >= gp.quicksum(V.z[(t, g, s, h)] for g in range(G) for s in range(S) for h in range(H)),
            name=f"c11_{t}",
        )

    if lifo_strengthening:
        # --- Constraint (15): bottom slot must be empty to receive a drop
        for t in range(1, R + 1):
            for s in range(S):
                m.addConstr(
                    gp.quicksum(V.y[(t, g, s, 0)] for g in range(G))
                    <= 1 - gp.quicksum(V.x[(t - 1, g, s, 0)] for g in range(G)),
                    name=f"c15_{t}_{s}",
                )
        # --- Constraint (16): drop-offs only onto the available top ------
        for t in range(1, R + 1):
            for s in range(S):
                for h in range(H - 1):
                    m.addConstr(
                        gp.quicksum(V.y[(t, g, s, h + 1)] for g in range(G))
                        <= gp.quicksum(V.x[(t - 1, g, s, h)] - V.x[(t - 1, g, s, h + 1)] for g in range(G)),
                        name=f"c16_{t}_{s}_{h}",
                    )
        # --- Constraint (17): pick-ups only from the current top ---------
        for t in range(1, R + 1):
            for s in range(S):
                for h in range(H - 1):
                    m.addConstr(
                        gp.quicksum(V.z[(t, g, s, h)] for g in range(G))
                        <= gp.quicksum(V.x[(t - 1, g, s, h)] - V.x[(t - 1, g, s, h + 1)] for g in range(G)),
                        name=f"c17_{t}_{s}_{h}",
                    )
        # --- Constraint (18): no immediate re-relocation of the same group
        for t in range(2, R + 1):
            for g in range(G):
                for s in range(S):
                    m.addConstr(
                        gp.quicksum(V.y[(t - 1, g, s, h)] + V.z[(t, g, s, h)] for h in range(H)) <= 1,
                        name=f"c18_{t}_{g}_{s}",
                    )
    else:
        # --- Constraint (4): no floating containers (kept only when the
        # stronger LIFO constraints above are switched off) ----------------
        for t in range(1, R + 1):
            for s in range(S):
                for h in range(H - 1):
                    m.addConstr(
                        gp.quicksum(V.x[(t, g, s, h)] for g in range(G))
                        >= gp.quicksum(V.x[(t, g, s, h + 1)] for g in range(G)),
                        name=f"c4_{t}_{s}_{h}",
                    )

    if flow_strengthening:
        # --- Constraint (20): no simultaneous pick-up and drop-off at s --
        for t in range(1, R + 1):
            for s in range(S):
                m.addConstr(
                    gp.quicksum(V.z[(t, g, s, h)] + V.y[(t, g, s, h)] for g in range(G) for h in range(H)) <= 1,
                    name=f"c20_{t}_{s}",
                )
        # --- Constraints (21)/(22): cross-stack matching ------------------
        for t in range(1, R + 1):
            for g in range(G):
                for s in range(S):
                    m.addConstr(
                        gp.quicksum(V.y[(t, g, s, h)] for h in range(H))
                        <= gp.quicksum(V.z[(t, g, r, h)] for r in range(S) if r != s for h in range(H)),
                        name=f"c21_{t}_{g}_{s}",
                    )
                    m.addConstr(
                        gp.quicksum(V.z[(t, g, s, h)] for h in range(H))
                        <= gp.quicksum(V.y[(t, g, r, h)] for r in range(S) if r != s for h in range(H)),
                        name=f"c22_{t}_{g}_{s}",
                    )

    # --- Objective (1): minimise total relocations ------------------------
    m.setObjective(
        gp.quicksum(V.y[(t, g, s, h)] for t in range(1, R + 1) for g in range(G) for s in range(S) for h in range(H)),
        GRB.MINIMIZE,
    )

    return m, V, stack_order
