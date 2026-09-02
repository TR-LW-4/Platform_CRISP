"""
r-BRP m1/m2 Gurobi formulation for DeMeloSilva2018RBRP.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from core.plan import Movement, RelocationPlan
from core.yard import Yard
from algorithms.CRP_R.solver_ip.common import (
    gurobi_on,
    priority_to_container_id,
    stack_keys,
)


# ================================================================ #
#  Helpers: yard → group-based configuration                         #
# ================================================================ #

def _yard_to_config(yard: Yard, S: int, H: int, G: int) -> Dict[Tuple[int,int,int], int]:
    """
    Return a dict  (g, s, h) → 1  for each occupied slot.
    g = priority (= group for unique-priority instances, 1..C).
    s = stack index (1..S), h = tier from bottom (1..H).
    """
    config: Dict[Tuple[int,int,int], int] = {}
    stack_list = sorted(yard.stacks.values(), key=lambda st: (st.bay, st.row))
    for s_idx, stack in enumerate(stack_list, 1):
        for h_idx, cont in enumerate(stack.containers, 1):  # h=1 = bottom
            g = cont.priority  # group = priority for unique-priority instances
            config[(g, s_idx, h_idx)] = 1
    return config


def _minmax_upper_bound(yard: Yard, C: int, S: int, T: int) -> int:
    """Caserta 2012 MinMax heuristic – quick upper bound on relocations."""
    stacks = [
        [c.priority for c in st.containers]
        for st in sorted(yard.stacks.values(), key=lambda st: (st.bay, st.row))
    ]
    height = [len(s) for s in stacks]

    def min_pri(si: int) -> int:
        return min(stacks[si]) if stacks[si] else C + 1

    min_p = [min_pri(s) for s in range(S)]
    n_reloc = 0

    for target_p in range(1, C + 1):
        ts = tp = None
        for si in range(S):
            for i, p in enumerate(stacks[si]):
                if p == target_p:
                    ts, tp = si, i
                    break
            if ts is not None:
                break
        if ts is None:
            continue
        while len(stacks[ts]) > tp + 1:
            n_reloc += 1
            r = stacks[ts][-1]
            good = [s for s in range(S) if s != ts and height[s] < T and min_p[s] > r]
            if good:
                dst = min(good, key=lambda s: min_p[s])
            else:
                avail = [s for s in range(S) if s != ts and height[s] < T]
                if not avail:
                    break
                dst = max(avail, key=lambda s: min_p[s])
            stacks[ts].pop()
            height[ts] -= 1
            stacks[dst].append(r)
            height[dst] += 1
            min_p[dst] = min_pri(dst)
            min_p[ts] = min_pri(ts)
        stacks[ts] = [p for p in stacks[ts] if p != target_p]
        height[ts] -= 1
        min_p[ts] = min_pri(ts)

    return n_reloc


def _plan_from_demelo_flow(
    yard: Yard,
    y,
    z,
    k,
    T: int,
    G: int,
    S: int,
    H: int,
    variant: str,
) -> RelocationPlan:
    """Turn de Melo y/z/k incumbents into relocate+retrieve moves."""
    keys = stack_keys(yard)
    pri_to_id = priority_to_container_id(yard)
    plan = RelocationPlan()
    ns = G
    for t in range(1, T + 1):
        y_hit = z_hit = k_hit = None
        for g in range(1, G + 1):
            for s in range(1, S + 1):
                for h in range(1, H + 1):
                    if gurobi_on(y[t, g, s, h]):
                        y_hit = (g, s, h)
                    if gurobi_on(z[t, g, s, h]):
                        z_hit = (g, s, h)
                    if variant == "m2" and gurobi_on(k[t, g, s, h]):
                        k_hit = (g, s, h)
        if variant != "m2":
            for g in range(1, G + 1):
                for n in range(1, ns + 1):
                    if (t, g, n) in k and gurobi_on(k[t, g, n]):
                        k_hit = (g, n)
                        break
                if k_hit is not None:
                    break
        if y_hit is not None and z_hit is not None:
            g, src, _h = z_hit
            _g, dst, _dh = y_hit
            plan.add(Movement(int(pri_to_id[g]), keys[src - 1], keys[dst - 1]))
        elif k_hit is not None:
            if variant == "m2":
                g, src, _h = k_hit
            else:
                g = k_hit[0]
                src = z_hit[1] if z_hit is not None else None
                if src is None:
                    continue
            plan.add(Movement(int(pri_to_id[g]), keys[src - 1], None))
    return plan


# ================================================================ #
#  r-BRP m1 solver                                                   #
# ================================================================ #

def _solve_rbrp_m1(
    yard: Yard,
    C: int,
    S: int,
    H: int,
    G: int,
    T_ub: int,
    time_limit_s: float,
    output_flag: int,
) -> Dict:
    """
    Restricted BRP model m1 (BRPm1 + Constraint 51).
    T_ub is the number of time steps = upper bound on relocations.
    """
    import gurobipy as gp
    from gurobipy import GRB

    T = T_ub  # total time steps = UB on relocations
    N = C     # total containers to retrieve

    cfg = _yard_to_config(yard, S, H, G)
    C_gsh: Dict[Tuple[int,int,int], int] = cfg

    m = gp.Model()
    m.Params.TimeLimit  = float(time_limit_s)
    m.Params.MIPGapAbs  = 0.99
    m.Params.OutputFlag = output_flag

    # Groups 1..G, Stacks 1..S, Heights 1..H, Time 0..T, Queue 1..N
    gs  = range(1, G + 1)
    ss  = range(1, S + 1)
    hs  = range(1, H + 1)
    ts  = range(1, T + 1)
    ns  = range(1, N + 1)

    # ── Variables ──────────────────────────────────────────────── #
    x = m.addVars([(t,g,s,h) for t in range(0,T+1) for g in gs for s in ss for h in hs],
                  vtype=GRB.BINARY, name="x")
    y = m.addVars([(t,g,s,h) for t in ts for g in gs for s in ss for h in hs],
                  vtype=GRB.BINARY, name="y")
    z = m.addVars([(t,g,s,h) for t in ts for g in gs for s in ss for h in hs],
                  vtype=GRB.BINARY, name="z")
    k = m.addVars([(t,g,n) for t in ts for g in gs for n in ns],
                  vtype=GRB.BINARY, name="k")
    w = m.addVars([(t,g,n) for t in range(0,T+1) for g in gs for n in ns],
                  vtype=GRB.BINARY, name="w")

    # ── Objective ─────────────────────────────────────────────── #
    m.setObjective(
        gp.quicksum(y[t,g,s,h] for t in ts for g in gs for s in ss for h in hs),
        GRB.MINIMIZE,
    )

    # ── Constraint (2): initialise x at t=0 ────────────────────── #
    for g in gs:
        for s in ss:
            for h in hs:
                m.addConstr(x[0,g,s,h] == C_gsh.get((g,s,h), 0))

    # ── Constraint (3): at most one group per slot ──────────────── #
    for t in range(0, T+1):
        for s in ss:
            for h in hs:
                m.addConstr(gp.quicksum(x[t,g,s,h] for g in gs) <= 1)

    # ── Constraint (4): no gaps in stacks ─────────────────────── #
    for t in range(0, T+1):
        for s in ss:
            for h in range(1, H):
                m.addConstr(
                    gp.quicksum(x[t,g,s,h] for g in gs) >=
                    gp.quicksum(x[t,g,s,h+1] for g in gs)
                )

    # ── Constraint (8): at most one y per step ─────────────────── #
    for t in ts:
        m.addConstr(
            gp.quicksum(y[t,g,s,h] for g in gs for s in ss for h in hs) <= 1
        )

    # ── Constraint (9): at most one z per step ─────────────────── #
    for t in ts:
        m.addConstr(
            gp.quicksum(z[t,g,s,h] for g in gs for s in ss for h in hs) <= 1
        )

    # ── Constraint (10): link x, y, z (flow) ──────────────────── #
    for t in ts:
        for g in gs:
            for s in ss:
                for h in hs:
                    m.addConstr(x[t,g,s,h] + z[t,g,s,h] >= x[t-1,g,s,h] + y[t,g,s,h])

    # ── Constraint (11): idle steps at end ────────────────────── #
    for t in range(2, T+1):
        m.addConstr(
            gp.quicksum(z[t-1,g,s,h] for g in gs for s in ss for h in hs) >=
            gp.quicksum(z[t,g,s,h]   for g in gs for s in ss for h in hs)
        )

    # ── Constraints (15)–(17): LIFO placement and pick-up ─────── #
    for t in ts:
        for s in ss:
            # (15): place only into empty bottom slot
            m.addConstr(
                gp.quicksum(y[t,g,s,1] for g in gs) <=
                1 - gp.quicksum(x[t-1,g,s,1] for g in gs)
            )
            for h in range(1, H):
                # (16): place only on top of occupied slot
                m.addConstr(
                    gp.quicksum(y[t,g,s,h+1] for g in gs) <=
                    gp.quicksum(x[t-1,g,s,h] for g in gs) -
                    gp.quicksum(x[t-1,g,s,h+1] for g in gs)
                )
                # (17): pick up only the topmost container
                m.addConstr(
                    gp.quicksum(z[t,g,s,h] for g in gs) <=
                    gp.quicksum(x[t-1,g,s,h] for g in gs) -
                    gp.quicksum(x[t-1,g,s,h+1] for g in gs)
                )

    # ── Constraint (19): slot occupancy (replaces C3) ─────────── #
    for t in ts:
        for s in ss:
            for h in hs:
                m.addConstr(
                    gp.quicksum(x[t,g,s,h] + z[t,g,s,h] for g in gs) <= 1
                )

    # ── Bay empty at end (Constraint 25) ──────────────────────── #
    m.addConstr(
        gp.quicksum(x[T,g,s,h] for g in gs for s in ss for h in hs) == 0
    )

    # ── Queue initialise (26) ──────────────────────────────────── #
    for g in gs:
        for n in ns:
            m.addConstr(w[0,g,n] == 0)

    # ── Queue final = Q_gn (27): unique priority → Q_{g,g}=1 ──── #
    # For unique priorities (G=N), container g → queue position g
    for g in gs:
        for n in ns:
            Q_gn = 1 if g == n else 0
            m.addConstr(w[T,g,n] == Q_gn)

    # ── At most one group per queue slot (28) ─────────────────── #
    for t in range(0, T+1):
        for n in ns:
            m.addConstr(gp.quicksum(w[t,g,n] for g in gs) <= 1)

    # ── At most one retrieval per step (29) ───────────────────── #
    for t in ts:
        m.addConstr(gp.quicksum(k[t,g,n] for g in gs for n in ns) <= 1)

    # ── Queue order preserved (30) ────────────────────────────── #
    for t in range(0, T+1):
        for n in range(1, N):
            m.addConstr(
                gp.quicksum(w[t,g,n] for g in gs) >=
                gp.quicksum(w[t,g,n+1] for g in gs)
            )

    # ── Queue consistency (31, 32) ────────────────────────────── #
    for t in ts:
        for g in gs:
            m.addConstr(
                gp.quicksum(w[t,g,n] for n in ns) ==
                gp.quicksum(w[t-1,g,n] + k[t,g,n] for n in ns)
            )
    for t in ts:
        for g in gs:
            for n in ns:
                m.addConstr(w[t,g,n] == w[t-1,g,n] + k[t,g,n])

    # ── Bay consistency with queue (33) ───────────────────────── #
    for t in ts:
        for g in gs:
            m.addConstr(
                gp.quicksum(x[t-1,g,s,h] for s in ss for h in hs) ==
                gp.quicksum(k[t,g,n] for n in ns) +
                gp.quicksum(x[t,g,s,h] for s in ss for h in hs)
            )

    # ── Link bay, relocation, retrieval (34) ──────────────────── #
    for t in ts:
        for g in gs:
            for s in ss:
                for h in hs:
                    m.addConstr(
                        x[t,g,s,h] + z[t,g,s,h] >=
                        x[t-1,g,s,h] + y[t,g,s,h]
                    )

    # ── z accounts for retrievals too (35) ────────────────────── #
    for t in ts:
        for g in gs:
            m.addConstr(
                gp.quicksum(y[t,g,s,h] for s in ss for h in hs) +
                gp.quicksum(k[t,g,n] for n in ns) ==
                gp.quicksum(z[t,g,s,h] for s in ss for h in hs)
            )

    # ── First N steps must each have exactly one z (36) ──────── #
    for t in range(1, N + 1):
        m.addConstr(
            gp.quicksum(z[t,g,s,h] for g in gs for s in ss for h in hs) == 1
        )

    # ── Constraint (51): Restricted assumption ─────────────────── #
    # Sum of z for groups j > g from any stack s, at time t ≤ w^{t-1}_{g,g} + x^t_{g,s,.}
    for t in ts:
        for g in range(1, G):  # g < G
            for s in ss:
                m.addConstr(
                    gp.quicksum(z[t,j,s,h] for j in range(g+1, G+1) for h in hs) <=
                    w[t-1,g,g] +
                    gp.quicksum(x[t,i,s,h] for i in range(1, g+1) for h in hs)
                )

    m.optimize()

    status = m.Status
    if m.SolCount == 0:
        return dict(obj=None, optimal=False, time_out=(status in (2,9,11)),
                    n_vars=m.NumVars, n_constrs=m.NumConstrs,
                    solve_time=m.Runtime, plan=None)
    obj = int(round(m.ObjVal))
    return dict(obj=obj, optimal=(status == 2),
                time_out=(status in (9,11)),
                n_vars=m.NumVars, n_constrs=m.NumConstrs,
                solve_time=m.Runtime,
                plan=_plan_from_demelo_flow(yard, y, z, k, T, G, S, H, "m1"))


# ================================================================ #
#  r-BRP m2 solver                                                   #
# ================================================================ #

def _solve_rbrp_m2(
    yard: Yard,
    C: int,
    S: int,
    H: int,
    G: int,
    T_ub: int,
    time_limit_s: float,
    output_flag: int,
) -> Dict:
    """
    Restricted BRP model m2 (BRPm2 + Constraint 52).
    More compact than m1; LP relaxation weaker but often faster.
    """
    import gurobipy as gp
    from gurobipy import GRB

    T = T_ub
    cfg = _yard_to_config(yard, S, H, G)
    C_gsh = cfg

    m = gp.Model()
    m.Params.TimeLimit  = float(time_limit_s)
    m.Params.MIPGapAbs  = 0.99
    m.Params.OutputFlag = output_flag

    gs = range(1, G + 1)
    ss = range(1, S + 1)
    hs = range(1, H + 1)
    ts = range(1, T + 1)

    # ── Variables ──────────────────────────────────────────────── #
    x = m.addVars([(t,g,s,h) for t in range(0,T+1) for g in gs for s in ss for h in hs],
                  vtype=GRB.BINARY, name="x")
    y = m.addVars([(t,g,s,h) for t in ts for g in gs for s in ss for h in hs],
                  vtype=GRB.BINARY, name="y")
    z = m.addVars([(t,g,s,h) for t in ts for g in gs for s in ss for h in hs],
                  vtype=GRB.BINARY, name="z")
    # k[t,g,s,h] = 1 if group g is retrieved from slot (s,h) at time t
    k = m.addVars([(t,g,s,h) for t in ts for g in gs for s in ss for h in hs],
                  vtype=GRB.BINARY, name="k")

    # ── Objective ─────────────────────────────────────────────── #
    m.setObjective(
        gp.quicksum(y[t,g,s,h] for t in ts for g in gs for s in ss for h in hs),
        GRB.MINIMIZE,
    )

    # ── Constraint (2): initialise ─────────────────────────────── #
    for g in gs:
        for s in ss:
            for h in hs:
                m.addConstr(x[0,g,s,h] == C_gsh.get((g,s,h), 0))

    # ── Constraint (4): no gaps ────────────────────────────────── #
    for t in range(0, T+1):
        for s in ss:
            for h in range(1, H):
                m.addConstr(
                    gp.quicksum(x[t,g,s,h] for g in gs) >=
                    gp.quicksum(x[t,g,s,h+1] for g in gs)
                )

    # ── Constraint (8): at most one y ──────────────────────────── #
    for t in ts:
        m.addConstr(
            gp.quicksum(y[t,g,s,h] for g in gs for s in ss for h in hs) <= 1
        )

    # ── Constraint (9): at most one z ──────────────────────────── #
    for t in ts:
        m.addConstr(
            gp.quicksum(z[t,g,s,h] for g in gs for s in ss for h in hs) <= 1
        )

    # ── Constraints (12)–(14): variable domains (already BINARY)

    # ── Constraints (15)–(17): LIFO ───────────────────────────── #
    for t in ts:
        for s in ss:
            m.addConstr(
                gp.quicksum(y[t,g,s,1] for g in gs) <=
                1 - gp.quicksum(x[t-1,g,s,1] for g in gs)
            )
            for h in range(1, H):
                m.addConstr(
                    gp.quicksum(y[t,g,s,h+1] for g in gs) <=
                    gp.quicksum(x[t-1,g,s,h] for g in gs) -
                    gp.quicksum(x[t-1,g,s,h+1] for g in gs)
                )
                m.addConstr(
                    gp.quicksum(z[t,g,s,h] for g in gs) <=
                    gp.quicksum(x[t-1,g,s,h] for g in gs) -
                    gp.quicksum(x[t-1,g,s,h+1] for g in gs)
                )

    # ── Constraint (20): no pick-up and drop-off same stack ───── #
    for t in ts:
        for s in ss:
            m.addConstr(
                gp.quicksum(z[t,g,s,h] + y[t,g,s,h] for g in gs for h in hs) <= 1
            )

    # ── Constraint (41): bay empty at end ─────────────────────── #
    m.addConstr(
        gp.quicksum(x[T,g,s,h] for g in gs for s in ss for h in hs) == 0
    )

    # ── Constraint (42): slot occupancy during retrieval ──────── #
    for t in ts:
        for s in ss:
            for h in hs:
                m.addConstr(
                    gp.quicksum(x[t,g,s,h] + k[t,g,s,h] for g in gs) <= 1
                )

    # ── Constraint (43): bay consistency with retrievals ──────── #
    for t in ts:
        for g in gs:
            m.addConstr(
                gp.quicksum(x[t-1,g,s,h] for s in ss for h in hs) ==
                gp.quicksum(x[t,g,s,h] + k[t,g,s,h] for s in ss for h in hs)
            )

    # ── Constraint (44): link z, k, y, x ─────────────────────── #
    for t in ts:
        for g in gs:
            for s in ss:
                for h in hs:
                    m.addConstr(
                        z[t,g,s,h] + k[t,g,s,h] ==
                        y[t,g,s,h] + x[t-1,g,s,h] - x[t,g,s,h]
                    )

    # ── Constraint (46): retrieval order (unique priorities) ───── #
    # Container g can only be retrieved after container g-1 was retrieved
    for t in ts:
        for g in range(2, G + 1):
            m.addConstr(
                gp.quicksum(k[t,g,s,h] for s in ss for h in hs) <=
                gp.quicksum(k[u,g-1,s,h] for u in range(1,t+1) for s in ss for h in hs)
            )

    # ── Constraint (47): no two same-stack retrievals same tier ── #
    for t in ts:
        for g in range(1, G):
            for s in ss:
                for h in range(1, H):
                    m.addConstr(
                        k[t,g,s,h] +
                        gp.quicksum(k[t,l,s,h+1] for l in range(g+1, G+1)) <= 1
                    )

    # ── Constraint (48): order within stack ───────────────────── #
    for t in ts:
        for g in range(2, G + 1):
            for s in ss:
                for h in range(2, H + 1):
                    m.addConstr(
                        k[t,g,s,h] <=
                        1 - gp.quicksum(x[t,l,s,h-1] for l in range(1,g))
                    )

    # ── Constraint (52): Restricted assumption for m2 ─────────── #
    for t in ts:
        for g in range(1, G):
            for s in ss:
                m.addConstr(
                    gp.quicksum(z[t,j,s,h] for j in range(g+1, G+1) for h in hs) <=
                    gp.quicksum(k[u,g,r,h] for u in range(1,t+1)
                                for r in ss for h in hs) +
                    gp.quicksum(x[t,i,s,h] for i in range(1, g+1) for h in hs)
                )

    m.optimize()

    status = m.Status
    if m.SolCount == 0:
        return dict(obj=None, optimal=False, time_out=(status in (2,9,11)),
                    n_vars=m.NumVars, n_constrs=m.NumConstrs,
                    solve_time=m.Runtime, plan=None)
    obj = int(round(m.ObjVal))
    return dict(obj=obj, optimal=(status == 2),
                time_out=(status in (9,11)),
                n_vars=m.NumVars, n_constrs=m.NumConstrs,
                solve_time=m.Runtime,
                plan=_plan_from_demelo_flow(yard, y, z, k, T, G, S, H, "m2"))
