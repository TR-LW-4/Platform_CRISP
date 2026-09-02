"""
Restricted-master MIP for VanBrinkVanDerZwaan2014BP.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

from .common import StackColumn


def solve_master_mip(
    columns_by_stack: Dict[int, Sequence[StackColumn]],
    priorities: Sequence[int],
    time_horizon: int,
    time_limit_s: float,
) -> Tuple[Optional[float], Optional[Dict[Tuple[int, int], float]]]:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        raise RuntimeError("Gurobi is required for this solver-assisted exact module.") from e

    m = gp.Model("prem_bp_master")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = max(0.1, float(time_limit_s))

    x = {}
    for s, cols in columns_by_stack.items():
        for j, col in enumerate(cols):
            if col.max_time_used <= time_horizon:
                x[(s, j)] = m.addVar(vtype=GRB.BINARY, name=f"x_{s}_{j}")

    if not x:
        return None, None

    m.setObjective(gp.quicksum(columns_by_stack[s][j].cost * var for (s, j), var in x.items()), GRB.MINIMIZE)

    # C3: one sequence per stack
    for s, cols in columns_by_stack.items():
        active = [x[(s, j)] for j in range(len(cols)) if (s, j) in x]
        if not active:
            return None, None
        m.addConstr(gp.quicksum(active) == 1, name=f"stack_{s}")

    # C2: at most one add at each time
    for t in range(1, time_horizon + 1):
        terms = []
        for (s, j), var in x.items():
            col = columns_by_stack[s][j]
            adds_at_t = sum(v for (p, tt), v in col.adds.items() if tt == t)
            if adds_at_t:
                terms.append(adds_at_t * var)
        if terms:
            m.addConstr(gp.quicksum(terms) <= 1, name=f"time_{t}")

    # C1: add - rem >= 0 for each (priority, time)
    for p in priorities:
        for t in range(1, time_horizon + 1):
            terms = []
            for (s, j), var in x.items():
                col = columns_by_stack[s][j]
                val = col.adds.get((p, t), 0) - col.rems.get((p, t), 0)
                if val:
                    terms.append(val * var)
            if terms:
                m.addConstr(gp.quicksum(terms) >= 0, name=f"flow_p{p}_t{t}")

    m.optimize()

    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        return None, None
    if m.SolCount <= 0:
        return None, None

    sol = {(s, j): float(var.X) for (s, j), var in x.items()}
    return float(m.ObjVal), sol

