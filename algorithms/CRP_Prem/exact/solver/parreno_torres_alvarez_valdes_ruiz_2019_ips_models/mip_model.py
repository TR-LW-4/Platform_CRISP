"""
IPS6 -- Parreno-Torres, Alvarez-Valdes & Ruiz (2019), Section 5.4, the
paper's own best-performing model ("we present the whole IPS6 model which
is our best proposal for solving the pre-marshalling problem").

Time is discretized into ``n_time_points`` time points (index t = 0 is the
given initial layout, index t = n_time_points - 1 must be fully sorted);
moves happen in the ``n_time_points - 1`` segments between consecutive time
points, at most one move per segment (matching every other exact PMP model
already embedded in this platform).

Variables (0-based indices in code; paper uses 1-based s/h/p/t):
  x[t,s,h,p]  = 1 if at time point t there is a priority-p container in
                slot (s,h), t = 0 .. n_time_points-1.
  w[t,s,h,p]  = 1 if, during segment t (transition t -> t+1), a priority-p
                container is moved OUT of slot (s,h), t = 0 .. n_time_points-2.
  z[t,s,h,p]  = 1 if, during segment t, a priority-p container is moved
                INTO slot (s,h).

Constraints implemented (paper numbering, Eq. 26-38):
  (26) objective: minimise total drop-offs (== total relocations)
  (27) per-priority flow balance within a segment (what leaves == what enters)
  (28) at most one pick-up / one drop-off per segment (K=1)
  (29) OPTIONAL ("earliest time" push): only meaningful when solving with a
       single, deliberately oversized T; the paper's own Section 6 notes
       this becomes unnecessary once the ascending iterative T-search is
       used (the default solve strategy here, see solve.py), so it is off
       by default.
  (30) bottom-slot guard: a drop can target the bottom tier of a stack only
       if it is empty in the state just before the segment.
  (31) sorted-final-layout rule, applied only at the last time point --
       identical role to Lee-Hsu's Eq.(20) / de Melo da Silva's Eq.(7).
  (32) flow link between consecutive states and the movement variables.
  (33) LIFO: a stack's tier h+1 can only be occupied (by inheritance, a
       drop, or staying put) if tier h is currently occupied; a pick-up
       from h or a drop into h+1 both require h to already hold a container
       (nothing can be lifted "through" an occupied slot above it).
  (34) strengthening: two consecutive occupied tiers cannot both be
       disturbed (picked from / dropped into) within the same segment.
  (35)/(36) transitive-move and same-priority-symmetry breaking across two
       consecutive segments (a container that just arrived at s cannot
       immediately leave s again; a priority group that just left s cannot
       have another member of the same priority arrive at s right after).
  (37)/(38) variable fixing in the very last segment: the top-priority
       group can never be the subject of the very last move (it never
       blocks anything below it), and non-bottom... see docstring in code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .core import Stacks

Key4 = Tuple[int, int, int, int]  # (t, s, h, p)


@dataclass
class ModelDims:
    n_stacks: int
    max_tiers: int
    priority_values: List[int]   # priority index p (0-based) -> actual priority value
    n_time_points: int           # paper's T (>= 1)

    @property
    def n_priorities(self) -> int:
        return len(self.priority_values)


@dataclass
class ModelVars:
    x: Dict[Key4, object] = field(default_factory=dict)
    w: Dict[Key4, object] = field(default_factory=dict)
    z: Dict[Key4, object] = field(default_factory=dict)


def build_ips6_model(
    stacks_init: Stacks,
    dims: ModelDims,
    time_limit_s: float,
    earliest_time_push: bool = False,
    transitive_move_breaking: bool = True,
    fix_last_segment_vars: bool = True,
    continuous_move_vars: bool = False,
):
    """
    Builds the Gurobi model for IPS6. Returns (model, vars, stack_order).

    ``continuous_move_vars``: Proposition 1 of the paper proves w/z can be
    declared as non-negative reals without losing integrality of the
    optimum (since x is binary); the paper itself keeps them binary anyway
    because it helps the iterative T-search detect infeasible T values
    faster, which is also this embedding's default.
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        raise RuntimeError("Gurobi is required for the Parreno-Torres et al. (2019) IPS6 model.") from e

    S = dims.n_stacks
    H = dims.max_tiers
    P = dims.n_priorities
    TP = dims.n_time_points
    n_segments = TP - 1
    stack_order = sorted(stacks_init.keys())
    prio_to_idx = {v: p for p, v in enumerate(dims.priority_values)}

    m = gp.Model("parreno_torres_2019_ips6")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = max(0.1, float(time_limit_s))

    move_vtype = GRB.CONTINUOUS if continuous_move_vars else GRB.BINARY

    V = ModelVars()

    for t in range(TP):
        for s in range(S):
            for h in range(H):
                for p in range(P):
                    V.x[(t, s, h, p)] = m.addVar(vtype=GRB.BINARY, name=f"x_{t}_{s}_{h}_{p}")

    for t in range(n_segments):
        for s in range(S):
            for h in range(H):
                for p in range(P):
                    kwargs = {} if continuous_move_vars else {}
                    V.w[(t, s, h, p)] = m.addVar(vtype=move_vtype, lb=0.0, ub=1.0, name=f"w_{t}_{s}_{h}_{p}")
                    V.z[(t, s, h, p)] = m.addVar(vtype=move_vtype, lb=0.0, ub=1.0, name=f"z_{t}_{s}_{h}_{p}")

    m.update()

    # C[s,h,p]: fixed constant from the initial layout (paper: "we
    # initialize variables x at time point 1 ... they are parameters").
    supply = {}
    for si, s in enumerate(stack_order):
        arr = stacks_init[s]
        for h in range(H):
            occ_p = prio_to_idx[arr[h]] if h < len(arr) else None
            for p in range(P):
                supply[(si, h, p)] = 1 if p == occ_p else 0

    for s in range(S):
        for h in range(H):
            for p in range(P):
                m.addConstr(V.x[(0, s, h, p)] == supply[(s, h, p)], name=f"init_{s}_{h}_{p}")

    # --- (27): per-priority flow balance within a segment ----------------
    for t in range(n_segments):
        for p in range(P):
            m.addConstr(
                gp.quicksum(V.w[(t, s, h, p)] for s in range(S) for h in range(H))
                == gp.quicksum(V.z[(t, s, h, p)] for s in range(S) for h in range(H)),
                name=f"c27_{t}_{p}",
            )

    # --- (28): at most one pick-up / drop-off per segment (K=1) ----------
    for t in range(n_segments):
        m.addConstr(
            gp.quicksum(V.w[(t, s, h, p)] for s in range(S) for h in range(H) for p in range(P)) <= 1,
            name=f"c28w_{t}",
        )
        m.addConstr(
            gp.quicksum(V.z[(t, s, h, p)] for s in range(S) for h in range(H) for p in range(P)) <= 1,
            name=f"c28z_{t}",
        )

    # --- (29) OPTIONAL: push moves to the earliest possible segment ------
    if earliest_time_push:
        for t in range(n_segments - 1):
            m.addConstr(
                gp.quicksum(V.w[(t + 1, s, h, p)] for s in range(S) for h in range(H) for p in range(P))
                <= gp.quicksum(V.w[(t, s, h, p)] for s in range(S) for h in range(H) for p in range(P)),
                name=f"c29_{t}",
            )

    # --- (30): bottom-slot guard ------------------------------------------
    for t in range(n_segments):
        for s in range(S):
            m.addConstr(
                gp.quicksum(V.z[(t, s, 0, p)] for p in range(P))
                <= 1 - gp.quicksum(V.x[(t, s, 0, p)] for p in range(P)),
                name=f"c30_{t}_{s}",
            )

    # --- (31): sorted final layout, applied only at the last time point --
    last = TP - 1
    for s in range(S):
        for h in range(H - 1):
            for p in range(P):
                m.addConstr(
                    gp.quicksum(V.x[(last, s, h + 1, k)] for k in range(p, P))
                    <= gp.quicksum(V.x[(last, s, h, k)] for k in range(p, P)),
                    name=f"c31_{s}_{h}_{p}",
                )

    # --- (32): flow link ----------------------------------------------------
    for t in range(n_segments):
        for s in range(S):
            for h in range(H):
                for p in range(P):
                    m.addConstr(
                        V.x[(t, s, h, p)] + V.z[(t, s, h, p)] == V.x[(t + 1, s, h, p)] + V.w[(t, s, h, p)],
                        name=f"c32_{t}_{s}_{h}_{p}",
                    )

    # --- (33): LIFO / no-floating-container --------------------------------
    for t in range(n_segments):
        for s in range(S):
            for h in range(H - 1):
                m.addConstr(
                    gp.quicksum(V.x[(t, s, h + 1, p)] for p in range(P))
                    + gp.quicksum(V.w[(t, s, h, p)] for p in range(P))
                    + gp.quicksum(V.z[(t, s, h + 1, p)] for p in range(P))
                    <= gp.quicksum(V.x[(t, s, h, p)] for p in range(P)),
                    name=f"c33_{t}_{s}_{h}",
                )

    # --- (34): two consecutive occupied tiers cannot both be disturbed ----
    for t in range(n_segments):
        for s in range(S):
            for h in range(H - 1):
                m.addConstr(
                    gp.quicksum(V.x[(t, s, h, p)] for p in range(P))
                    + gp.quicksum(V.x[(t, s, h + 1, p)] for p in range(P))
                    + gp.quicksum(V.w[(t, s, h, p)] for p in range(P))
                    + gp.quicksum(V.z[(t, s, h + 1, p)] for p in range(P))
                    + gp.quicksum(V.z[(t, s, h, p)] for p in range(P))
                    <= 2,
                    name=f"c34_{t}_{s}_{h}",
                )

    if transitive_move_breaking:
        # --- (35): no transitive moves across two consecutive segments ---
        for t in range(n_segments - 1):
            for s in range(S):
                m.addConstr(
                    gp.quicksum(V.z[(t, s, h, p)] for h in range(H) for p in range(P))
                    + gp.quicksum(V.w[(t + 1, s, h, p)] for h in range(H) for p in range(P))
                    <= 1,
                    name=f"c35_{t}_{s}",
                )
        # --- (36): same-priority symmetry breaking ------------------------
        for t in range(n_segments - 1):
            for s in range(S):
                for p in range(P):
                    m.addConstr(
                        gp.quicksum(V.w[(t, s, h, p)] for h in range(H))
                        + gp.quicksum(V.z[(t + 1, s, h, p)] for h in range(H))
                        <= 1,
                        name=f"c36_{t}_{s}_{p}",
                    )

    if fix_last_segment_vars and n_segments > 0:
        last_seg = n_segments - 1
        # --- (37): the top-priority group can never be the last move -----
        for s in range(S):
            for h in range(H):
                m.addConstr(V.w[(last_seg, s, h, 0)] == 0, name=f"c37w_{s}_{h}")
                m.addConstr(V.z[(last_seg, s, h, 0)] == 0, name=f"c37z_{s}_{h}")
        # --- (38): no non-top-priority pick-up from the bottom tier in the
        # last segment ------------------------------------------------------
        for s in range(S):
            for p in range(1, P):
                m.addConstr(V.w[(last_seg, s, 0, p)] == 0, name=f"c38_{s}_{p}")

    # --- Objective (26): minimise total relocations ------------------------
    m.setObjective(
        gp.quicksum(V.z[(t, s, h, p)] for t in range(n_segments) for s in range(S) for h in range(H) for p in range(P)),
        GRB.MINIMIZE,
    )

    return m, V, stack_order
