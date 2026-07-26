"""
Section 8 extensions from Parreno-Torres, Alvarez-Valdes & Ruiz (2019). All
four are optional add-ons on top of the IPS6 model that only touch the last
time point's ``x`` variables (or the objective with an infinitesimally
small extra term), so none of them can change the optimal number of
relocations -- they only break ties among equally-good solutions.

* Eq. (40)/(41): cap the container-count difference between adjacent
  stacks in the final layout (avoid "one empty stack next to a full one").
* Eq. (42)/(43): forbid an empty or a completely full stack in the final
  layout.
* Objective term (h_max - h_min)/(H+1): prefer a more balanced final
  layout without ever changing the optimal move count, since this term is
  always < 1 (Section 8.3).
* Objective term -sum(g_shp)/sum(n_p): prefer same-priority containers
  stacked directly on top of each other in the final layout, again always
  < 1 in magnitude (Section 8.4).
"""

from __future__ import annotations

from typing import List

from .mip_model import ModelDims, ModelVars


def add_balance_constraint(m, V: ModelVars, dims: ModelDims, kappa: int) -> None:
    """Eq. (40)/(41): |containers(s) - containers(s+1)| <= kappa in the
    final layout, for adjacent stacks s, s+1."""
    S = dims.n_stacks
    H = dims.max_tiers
    P = dims.n_priorities
    last = dims.n_time_points - 1

    def height_expr(s: int):
        return sum(V.x[(last, s, h, p)] for h in range(H) for p in range(P))

    for s in range(S - 1):
        m.addConstr(height_expr(s) - height_expr(s + 1) <= kappa, name=f"c40_{s}")
        m.addConstr(height_expr(s) - height_expr(s + 1) >= -kappa, name=f"c41_{s}")


def add_no_empty_or_full_stacks(m, V: ModelVars, dims: ModelDims) -> None:
    """Eq. (42)/(43): no stack is empty or completely full in the final
    layout."""
    S = dims.n_stacks
    H = dims.max_tiers
    P = dims.n_priorities
    last = dims.n_time_points - 1

    for s in range(S):
        height = sum(V.x[(last, s, h, p)] for h in range(H) for p in range(P))
        m.addConstr(height >= 1, name=f"c42_{s}")
        m.addConstr(height <= H - 1, name=f"c43_{s}")


def add_stability_bonus(m, V: ModelVars, dims: ModelDims) -> None:
    """Section 8.3: add (h_max - h_min)/(H+1) to the objective to prefer a
    more balanced final layout among equally-good (minimum relocation)
    solutions. The term is always < 1 so it never changes the optimal
    number of relocations."""
    import gurobipy as gp
    from gurobipy import GRB

    S = dims.n_stacks
    H = dims.max_tiers
    P = dims.n_priorities
    last = dims.n_time_points - 1

    h_min = m.addVar(lb=0.0, ub=H, vtype=GRB.CONTINUOUS, name="h_min")
    h_max = m.addVar(lb=0.0, ub=H, vtype=GRB.CONTINUOUS, name="h_max")
    for s in range(S):
        height = gp.quicksum(V.x[(last, s, h, p)] for h in range(H) for p in range(P))
        m.addConstr(height <= h_max, name=f"c44_{s}")
        m.addConstr(height >= h_min, name=f"c45_{s}")

    m.setObjective(m.getObjective() + (h_max - h_min) / (H + 1), GRB.MINIMIZE)


def add_same_priority_bonus(m, V: ModelVars, dims: ModelDims, total_containers: int) -> None:
    """Section 8.4: reward same-priority containers stacked directly on
    top of each other in the final layout. The reward is always < 1 in
    magnitude so it never changes the optimal number of relocations."""
    import gurobipy as gp
    from gurobipy import GRB

    S = dims.n_stacks
    H = dims.max_tiers
    P = dims.n_priorities
    last = dims.n_time_points - 1

    if total_containers <= 0:
        return

    g_vars: List = []
    for s in range(S):
        for h in range(1, H):
            for p in range(P):
                g = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"g_{s}_{h}_{p}")
                m.addConstr(g <= V.x[(last, s, h, p)], name=f"c46_{s}_{h}_{p}")
                m.addConstr(
                    g <= 1 - (V.x[(last, s, h, p)] - V.x[(last, s, h - 1, p)]),
                    name=f"c47_{s}_{h}_{p}",
                )
                g_vars.append(g)

    m.setObjective(m.getObjective() - gp.quicksum(g_vars) / total_containers, GRB.MINIMIZE)
