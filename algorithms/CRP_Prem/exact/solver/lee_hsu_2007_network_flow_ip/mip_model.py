"""
Time-expanded multi-commodity flow network and basic model (Lee & Hsu 2007,
Sections 3.1-3.2), constraints (1)-(26), plus the two relaxations the paper
itself recommends for tractability (Section 5):

* multi-move relaxation: constraint (4) (<=1 movement arc per segment) is
  replaced by a cap ``max_moves_per_segment`` = K, and constraints
  (3)/(5)/(21) are replaced by the paper's (33)/(34) so several containers
  may be lifted from / placed onto stacks within one time segment.
* cycle-breaking constraints (31)/(32): forbid length-2 and length-3
  movement cycles from forming within a single segment. Longer cycles (if
  K >= 4) can still occur and are broken in a post-processing step, see
  ``ordering.py``.

Indices below are all 0-based Python indices; the docstrings note the
corresponding 1-based paper notation. Time point t=0 is the initial layout,
time point t=T-1 is the final layout; "segment t" (0 <= t <= T-2) is the
transition between time point t and time point t+1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .core import Stacks

Key4 = Tuple[int, int, int, int]  # (t, s, h, c)
KeyR = Tuple[int, int, int, int]  # (t, s, z, c)


@dataclass
class ModelDims:
    n_stacks: int
    max_tiers: int
    type_values: List[int]     # commodity index c -> actual priority value
    n_time_points: int         # T (>= 2)

    @property
    def n_types(self) -> int:
        return len(self.type_values)


@dataclass
class ModelVars:
    cu: Dict[Key4, object] = field(default_factory=dict)
    cd: Dict[Key4, object] = field(default_factory=dict)
    ci: Dict[Key4, object] = field(default_factory=dict)
    co: Dict[Key4, object] = field(default_factory=dict)
    cr: Dict[KeyR, object] = field(default_factory=dict)


def build_basic_model(
    stacks_init: Stacks,
    dims: ModelDims,
    time_limit_s: float,
    max_moves_per_segment: int = 1,
    cycle_breaking_level: int = 0,
):
    """
    Builds the Gurobi model for the basic network-flow formulation.

    ``max_moves_per_segment`` == 1 reproduces the paper's literal
    constraint (4). Values > 1 apply the (33)/(34) relaxation instead of
    (3)/(5)/(21) so multiple containers may move within one segment.

    ``cycle_breaking_level`` in {0, 2, 3}: 0 disables (31)/(32) entirely
    (only sensible when max_moves_per_segment == 1, since a single move can
    never form a cycle); 2 adds the 2-stack cycle-breaking constraint (31);
    3 additionally adds the 3-stack constraint (32).

    Returns (model, gurobi_vars, stack_order) where ``stack_order`` maps
    the model's 0-based stack index back to the caller's stack keys.
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        raise RuntimeError("Gurobi is required for the Lee & Hsu (2007) network-flow model.") from e

    S = dims.n_stacks
    H = dims.max_tiers
    C = dims.n_types
    T = dims.n_time_points
    stack_order = sorted(stacks_init.keys())
    type_to_idx = {v: c for c, v in enumerate(dims.type_values)}

    m = gp.Model("lee_hsu_2007_prem")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = max(0.1, float(time_limit_s))

    V = ModelVars()

    for t in range(T - 1):
        for s in range(S):
            for h in range(H - 1):
                for c in range(C):
                    V.cu[(t, s, h, c)] = m.addVar(vtype=GRB.BINARY, name=f"cu_{t}_{s}_{h}_{c}")

    for t in range(1, T):
        for s in range(S):
            for h in range(1, H):
                for c in range(C):
                    V.cd[(t, s, h, c)] = m.addVar(vtype=GRB.BINARY, name=f"cd_{t}_{s}_{h}_{c}")

    for t in range(T):
        for s in range(S):
            for h in range(H):
                for c in range(C):
                    V.ci[(t, s, h, c)] = m.addVar(vtype=GRB.BINARY, name=f"ci_{t}_{s}_{h}_{c}")

    for t in range(T - 1):
        for s in range(S):
            for h in range(H):
                for c in range(C):
                    V.co[(t, s, h, c)] = m.addVar(vtype=GRB.BINARY, name=f"co_{t}_{s}_{h}_{c}")

    for t in range(T - 1):
        for s in range(S):
            for z in range(S):
                if z == s:
                    continue
                for c in range(C):
                    V.cr[(t, s, z, c)] = m.addVar(vtype=GRB.BINARY, name=f"cr_{t}_{s}_{z}_{c}")

    m.update()

    # cSUPPLY[s,h,c]: fixed constant from the initial layout.
    supply = {}
    for si, s in enumerate(stack_order):
        arr = stacks_init[s]
        for h in range(H):
            occ_c = type_to_idx[arr[h]] if h < len(arr) else None
            for c in range(C):
                supply[(si, h, c)] = 1 if c == occ_c else 0

    # --- Constraint (1): no floating containers -------------------------
    for t in range(T):
        for s in range(S):
            for h in range(H - 1):
                m.addConstr(
                    gp.quicksum(V.ci[(t, s, h, c)] for c in range(C))
                    >= gp.quicksum(V.ci[(t, s, h + 1, c)] for c in range(C)),
                    name=f"c1_{t}_{s}_{h}",
                )

    # --- Constraints (2)/(3) or the (33)/(34) multi-move relaxation -----
    multi_move = max_moves_per_segment > 1
    if not multi_move:
        for t in range(T - 1):
            for s in range(S):
                for h in range(max(H - 2, 0)):
                    for c in range(C):
                        m.addConstr(
                            V.cu[(t, s, h, c)] <= V.cu[(t, s, h + 1, c)],
                            name=f"c2_{t}_{s}_{h}_{c}",
                        )
        for t in range(T - 1):
            for s in range(S):
                for h in range(H - 1):
                    m.addConstr(
                        gp.quicksum(V.ci[(t, s, h + 1, c)] for c in range(C))
                        + gp.quicksum(V.cu[(t, s, h, c)] for c in range(C))
                        <= 1,
                        name=f"c3_{t}_{s}_{h}",
                    )
    else:
        # (33): at most ``h`` (1-based) containers may be "in transit"
        # upward through slot h at once -- a container can only be lifted
        # above as many containers as physically sit below the slot.
        big_m = H + 1
        for t in range(T - 1):
            for s in range(S):
                for h in range(H - 1):
                    m.addConstr(
                        gp.quicksum(V.cu[(t, s, h, c)] for c in range(C)) <= h + 1,
                        name=f"c33_{t}_{s}_{h}",
                    )
        # (34): replaces (3) -- allows lifting a container out from under
        # others within the same segment (multiple exits per stack).
        for t in range(T - 1):
            for s in range(S):
                for h in range(H - 1):
                    m.addConstr(
                        big_m * gp.quicksum(V.co[(t, s, h + 1, c)] for c in range(C))
                        + gp.quicksum(V.cu[(t, s, h, c)] for c in range(C))
                        <= big_m,
                        name=f"c34_{t}_{s}_{h}",
                    )

    # --- Constraint (4)/relaxed cap: moves per segment -------------------
    for t in range(T - 1):
        m.addConstr(
            gp.quicksum(V.cr[(t, s, z, c)] for s in range(S) for z in range(S) if z != s for c in range(C))
            <= max_moves_per_segment,
            name=f"c4_{t}",
        )

    # --- Constraints (5)-(9): at most one unit of flow per arc -----------
    if not multi_move:
        for t in range(T - 1):
            for s in range(S):
                for h in range(H - 1):
                    m.addConstr(gp.quicksum(V.cu[(t, s, h, c)] for c in range(C)) <= 1, name=f"c5_{t}_{s}_{h}")
    for t in range(T):
        for s in range(S):
            for h in range(H):
                m.addConstr(gp.quicksum(V.ci[(t, s, h, c)] for c in range(C)) <= 1, name=f"c6_{t}_{s}_{h}")
    for t in range(1, T):
        for s in range(S):
            for h in range(1, H):
                m.addConstr(gp.quicksum(V.cd[(t, s, h, c)] for c in range(C)) <= 1, name=f"c7_{t}_{s}_{h}")
    for t in range(T - 1):
        for s in range(S):
            for h in range(H):
                m.addConstr(gp.quicksum(V.co[(t, s, h, c)] for c in range(C)) <= 1, name=f"c8_{t}_{s}_{h}")
    for t in range(T - 1):
        for s in range(S):
            for z in range(S):
                if z == s:
                    continue
                m.addConstr(gp.quicksum(V.cr[(t, s, z, c)] for c in range(C)) <= 1, name=f"c9_{t}_{s}_{z}")

    # --- Constraints (10)/(11): initial layout fixes ci at t=0 -----------
    for s in range(S):
        for h in range(H):
            for c in range(C):
                m.addConstr(V.ci[(0, s, h, c)] == supply[(s, h, c)], name=f"c10_11_{s}_{h}_{c}")

    # --- Constraints (13)-(15) & (17)-(19): flow conservation ------------
    for t in range(1, T):
        for s in range(S):
            for h in range(H):
                is_top = h == H - 1
                is_bottom = h == 0
                for c in range(C):
                    inflow = V.co[(t - 1, s, h, c)]
                    if not is_top:
                        inflow = inflow + V.cd[(t, s, h + 1, c)]
                    else:
                        inflow = inflow + gp.quicksum(
                            V.cr[(t - 1, z, s, c)] for z in range(S) if z != s
                        )
                    outflow = V.ci[(t, s, h, c)]
                    if not is_bottom:
                        outflow = outflow + V.cd[(t, s, h, c)]
                    m.addConstr(inflow == outflow, name=f"c1315_{t}_{s}_{h}_{c}")

    # --- Constraint (16): at most one arrival at a stack's top -----------
    for t in range(1, T):
        for s in range(S):
            m.addConstr(
                gp.quicksum(V.co[(t - 1, s, H - 1, c)] for c in range(C))
                + gp.quicksum(V.cr[(t - 1, z, s, c)] for z in range(S) if z != s for c in range(C))
                <= 1,
                name=f"c16_{t}_{s}",
            )

    for t in range(T - 1):
        for s in range(S):
            for h in range(H):
                is_top = h == H - 1
                is_bottom = h == 0
                for c in range(C):
                    inflow = V.ci[(t, s, h, c)]
                    if not is_bottom:
                        inflow = inflow + V.cu[(t, s, h - 1, c)]
                    outflow = V.co[(t, s, h, c)]
                    if not is_top:
                        outflow = outflow + V.cu[(t, s, h, c)]
                    else:
                        outflow = outflow + gp.quicksum(
                            V.cr[(t, s, z, c)] for z in range(S) if z != s
                        )
                    m.addConstr(inflow == outflow, name=f"c1719_{t}_{s}_{h}_{c}")

    # --- Constraint (20): correct stacking order in the final layout -----
    values = dims.type_values
    for s in range(S):
        for h in range(H - 1):
            m.addConstr(
                gp.quicksum(values[c] * V.ci[(T - 1, s, h, c)] for c in range(C))
                >= gp.quicksum(values[c] * V.ci[(T - 1, s, h + 1, c)] for c in range(C)),
                name=f"c20_{s}_{h}",
            )

    # --- Cycle-breaking constraints (31)/(32), only meaningful when
    # multiple moves per segment are allowed ------------------------------
    if multi_move and cycle_breaking_level >= 2:
        for t in range(T - 1):
            for s in range(S):
                for z in range(s + 1, S):
                    m.addConstr(
                        gp.quicksum(V.cr[(t, s, z, c)] for c in range(C))
                        + gp.quicksum(V.cr[(t, z, s, c)] for c in range(C))
                        <= 1,
                        name=f"c31_{t}_{s}_{z}",
                    )
    if multi_move and cycle_breaking_level >= 3:
        for t in range(T - 1):
            for q in range(S):
                for s in range(S):
                    if s == q:
                        continue
                    for z in range(S):
                        if z == q or z == s:
                            continue
                        m.addConstr(
                            gp.quicksum(V.cr[(t, s, z, c)] for c in range(C))
                            + gp.quicksum(V.cr[(t, z, q, c)] for c in range(C))
                            + gp.quicksum(V.cr[(t, q, s, c)] for c in range(C))
                            <= 2,
                            name=f"c32_{t}_{q}_{s}_{z}",
                        )

    # --- Objective (26): minimize total container movements --------------
    m.setObjective(
        gp.quicksum(
            V.cr[(t, s, z, c)]
            for t in range(T - 1)
            for s in range(S)
            for z in range(S)
            if z != s
            for c in range(C)
        ),
        GRB.MINIMIZE,
    )

    return m, V, stack_order
