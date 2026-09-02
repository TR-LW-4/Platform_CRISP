"""
Robust master MIP for BogeGoerigkKnust2020RobustPMP.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from .core import Stacks


@dataclass
class MasterResult:
    target_stacks: Stacks
    upper_obj: int


def solve_upper_bound_master(
    item_classes: Sequence[int],
    n_stacks: int,
    max_tiers: int,
    delta: int,
) -> MasterResult:
    """
    MIP for BIbar^rob minimization (paper-inspired upper-bound model).
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        raise RuntimeError("Gurobi is required for Boge-Goerigk-Knust (2020) model.") from e

    n_items = len(item_classes)
    items = list(range(n_items))
    stacks = list(range(n_stacks))

    # J_i: items whose classes can make i potentially badly placed under U_delta.
    j_sets: Dict[int, List[int]] = {}
    for i in items:
        ci = int(item_classes[i])
        js: List[int] = []
        for j in items:
            if i == j:
                continue
            cj = int(item_classes[j])
            if ci < cj <= ci + delta:
                js.append(j)
        j_sets[i] = js

    m = gp.Model("robust_prem_2020_upper")
    m.Params.OutputFlag = 0

    alpha = {(i, q): m.addVar(vtype=GRB.BINARY, name=f"a_{i}_{q}") for i in items for q in stacks}
    beta = {(i, q): m.addVar(vtype=GRB.BINARY, name=f"b_{i}_{q}") for i in items for q in stacks}

    m.setObjective(gp.quicksum(beta[i, q] for i in items for q in stacks), GRB.MINIMIZE)

    # Capacity per stack.
    for q in stacks:
        m.addConstr(
            gp.quicksum(alpha[i, q] + beta[i, q] for i in items) <= max_tiers,
            name=f"cap_{q}",
        )

    # Each item assigned exactly once.
    for i in items:
        m.addConstr(
            gp.quicksum(alpha[i, q] + beta[i, q] for q in stacks) == 1,
            name=f"assign_{i}",
        )

    # Robust upper-bound conflict constraints.
    for i in items:
        for j in j_sets[i]:
            for q in stacks:
                m.addConstr(
                    alpha[i, q] + alpha[j, q] + beta[j, q] <= 1,
                    name=f"conf_{i}_{j}_{q}",
                )

    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Upper-bound robust MIP did not solve optimally (status={m.Status}).")

    assigned: Dict[int, List[int]] = {q: [] for q in stacks}
    for i in items:
        chosen_q = None
        for q in stacks:
            if (alpha[i, q].X + beta[i, q].X) > 0.5:
                chosen_q = q
                break
        if chosen_q is None:
            continue
        assigned[chosen_q].append(int(item_classes[i]))

    # Build an explicit stack layout bottom..top.
    # Sorting by class descending makes nominal stacking feasible.
    for q in stacks:
        assigned[q].sort(reverse=True)
        if len(assigned[q]) > max_tiers:
            assigned[q] = assigned[q][:max_tiers]

    return MasterResult(target_stacks=assigned, upper_obj=int(round(m.ObjVal)))
