"""
DeMeloSilva2018BRPGrouped
<2018> <exact> <grouped> <single-bay> <CRP-D>
Time-indexed MIP for grouped / duplicate-priority BRP
variant --- m1 --- Formulation: m1 (T=UB+N) or m2 (T=UB)
time_limit_s --- 3600 --- Gurobi time limit (s)

------------------------------- Reference --------------------------------
M. de Melo da Silva, S. Toulouse, R. Wolfler Calvo,
"A new effective unified model for solving the Pre-marshalling and
 Block Relocation Problems",
European Journal of Operational Research 271 (2018) 40–56.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.yard import Yard


# ================================================================ #
#  Helpers                                                           #
# ================================================================ #

def _yard_to_config_grouped(
    yard: Yard, S: int, H: int
) -> Tuple[Dict[Tuple[int, int, int], int], Dict[int, int]]:
    """
    Return (config, group_sizes).

    config      : {(g, s, h): 1} for each occupied slot (1-indexed).
    group_sizes : {g: count_of_containers_in_group_g}.
    """
    config: Dict[Tuple[int, int, int], int] = {}
    group_sizes: Dict[int, int] = {}
    stack_list = sorted(yard.stacks.values(), key=lambda st: (st.bay, st.row))
    for s_idx, stack in enumerate(stack_list, 1):
        for h_idx, cont in enumerate(stack.containers, 1):
            g = cont.priority
            config[(g, s_idx, h_idx)] = 1
            group_sizes[g] = group_sizes.get(g, 0) + 1
    return config, group_sizes


def _cumulative_positions(G: int, group_sizes: Dict[int, int]) -> Dict[int, int]:
    """
    Return cum[g] = last queue position (1-indexed) allocated to group g.

    Group g occupies consecutive queue positions  cum[g-1]+1 .. cum[g].
    cum[0] = 0.
    """
    cum: Dict[int, int] = {0: 0}
    for g in range(1, G + 1):
        cum[g] = cum[g - 1] + group_sizes.get(g, 0)
    return cum


def _greedy_ub_grouped(yard: Yard, G: int, S: int, H: int) -> int:
    """
    Simple greedy upper bound for grouped BRP (duplicate priorities).

    Processes groups 1, 2, ..., G in order.  For each group, iteratively
    relocates whatever blocker sits on top of a stack that contains the
    current target group, then retrieves exposed target containers.
    Returns the total number of relocations performed.
    """
    stacks = [
        [c.priority for c in st.containers]
        for st in sorted(yard.stacks.values(), key=lambda st: (st.bay, st.row))
    ]
    height = [len(s) for s in stacks]
    n_reloc = 0

    for tg in range(1, G + 1):
        while any(tg in stacks[si] for si in range(S)):
            retrieved = False
            for si in range(S):
                if stacks[si] and stacks[si][-1] == tg:
                    stacks[si].pop()
                    height[si] -= 1
                    retrieved = True
            if retrieved:
                continue
            # Must relocate a blocker
            moved = False
            for si in range(S):
                if tg in stacks[si] and stacks[si] and stacks[si][-1] != tg:
                    blocker = stacks[si][-1]
                    dst = next(
                        (d for d in range(S) if d != si and height[d] < H),
                        None,
                    )
                    if dst is not None:
                        stacks[si].pop()
                        height[si] -= 1
                        stacks[dst].append(blocker)
                        height[dst] += 1
                        n_reloc += 1
                        moved = True
                        break
            if not moved:
                break

    return n_reloc


# ================================================================ #
#  Shared model builder (Eq. 2, 4, 8, 9, 15-17, 20)               #
# ================================================================ #

def _build_common_constraints(m, x, y, z, cfg, gs, ss, hs, ts, T, H, G, _N):
    """Build constraints shared by BRP m1 and m2 (Eq 2, 4, 8, 9, 15-17, 20)."""
    import gurobipy as gp

    # (2) initial bay state
    for g in gs:
        for s in ss:
            for h in hs:
                m.addConstr(x[0, g, s, h] == cfg.get((g, s, h), 0))

    # (4) no gaps (column fills from bottom)
    for t in range(0, T + 1):
        for s in ss:
            for h in range(1, H):
                m.addConstr(
                    gp.quicksum(x[t, g, s, h]     for g in gs) >=
                    gp.quicksum(x[t, g, s, h + 1] for g in gs)
                )

    # (8) at most one drop-off per step
    for t in ts:
        m.addConstr(
            gp.quicksum(y[t, g, s, h] for g in gs for s in ss for h in hs) <= 1
        )

    # (9) at most one pick-up per step
    for t in ts:
        m.addConstr(
            gp.quicksum(z[t, g, s, h] for g in gs for s in ss for h in hs) <= 1
        )

    # (15)-(17) LIFO
    for t in ts:
        for s in ss:
            m.addConstr(
                gp.quicksum(y[t, g, s, 1] for g in gs) <=
                1 - gp.quicksum(x[t - 1, g, s, 1] for g in gs)
            )
            for h in range(1, H):
                m.addConstr(
                    gp.quicksum(y[t, g, s, h + 1] for g in gs) <=
                    gp.quicksum(x[t - 1, g, s, h]     for g in gs) -
                    gp.quicksum(x[t - 1, g, s, h + 1] for g in gs)
                )
                m.addConstr(
                    gp.quicksum(z[t, g, s, h] for g in gs) <=
                    gp.quicksum(x[t - 1, g, s, h]     for g in gs) -
                    gp.quicksum(x[t - 1, g, s, h + 1] for g in gs)
                )

    # (20) no pick-up and drop-off in same stack same step
    for t in ts:
        for s in ss:
            m.addConstr(
                gp.quicksum(
                    z[t, g, s, h] + y[t, g, s, h] for g in gs for h in hs
                ) <= 1
            )


# ================================================================ #
#  BRP m1 — unrestricted, grouped priorities                        #
# ================================================================ #

def _solve_brp_m1_grouped(
    yard: Yard,
    N: int, S: int, H: int, G: int,
    group_sizes: Dict[int, int],
    T_ub: int,
    time_limit_s: float,
    output_flag: int,
) -> Dict:
    """
    Unrestricted BRP m1 with grouped priorities.

    Key change vs. unique-priority version: Constraint (27) uses
    Q_{g,n} = 1  iff  cum[g-1] < n <= cum[g],  where cum[g] is the
    last queue position allocated to group g.

    T = T_ub + N  (relocation steps + N retrieval steps).
    """
    import gurobipy as gp
    from gurobipy import GRB

    T = T_ub + N
    cfg, _ = _yard_to_config_grouped(yard, S, H)
    cum = _cumulative_positions(G, group_sizes)

    m = gp.Model()
    m.Params.TimeLimit  = float(time_limit_s)
    m.Params.MIPGapAbs  = 0.99
    m.Params.OutputFlag = output_flag

    gs = range(1, G + 1)
    ss = range(1, S + 1)
    hs = range(1, H + 1)
    ts = range(1, T + 1)
    ns = range(1, N + 1)

    x = m.addVars(
        [(t, g, s, h) for t in range(0, T + 1) for g in gs for s in ss for h in hs],
        vtype=GRB.BINARY, name="x",
    )
    y = m.addVars(
        [(t, g, s, h) for t in ts for g in gs for s in ss for h in hs],
        vtype=GRB.BINARY, name="y",
    )
    z = m.addVars(
        [(t, g, s, h) for t in ts for g in gs for s in ss for h in hs],
        vtype=GRB.BINARY, name="z",
    )
    k = m.addVars(
        [(t, g, n) for t in ts for g in gs for n in ns],
        vtype=GRB.BINARY, name="k",
    )
    w = m.addVars(
        [(t, g, n) for t in range(0, T + 1) for g in gs for n in ns],
        vtype=GRB.BINARY, name="w",
    )

    m.setObjective(
        gp.quicksum(y[t, g, s, h] for t in ts for g in gs for s in ss for h in hs),
        GRB.MINIMIZE,
    )

    _build_common_constraints(m, x, y, z, cfg, gs, ss, hs, ts, T, H, G, N)

    # (10) flow balance
    for t in ts:
        for g in gs:
            for s in ss:
                for h in hs:
                    m.addConstr(
                        x[t, g, s, h] + z[t, g, s, h] >=
                        x[t - 1, g, s, h] + y[t, g, s, h]
                    )

    # (11) retrievals are non-increasing (idle steps at the end)
    for t in range(2, T + 1):
        m.addConstr(
            gp.quicksum(z[t - 1, g, s, h] for g in gs for s in ss for h in hs) >=
            gp.quicksum(z[t,     g, s, h] for g in gs for s in ss for h in hs)
        )

    # (25) bay empty at end
    m.addConstr(
        gp.quicksum(x[T, g, s, h] for g in gs for s in ss for h in hs) == 0
    )

    # (26) queue empty at t=0
    for g in gs:
        for n in ns:
            m.addConstr(w[0, g, n] == 0)

    # (27) queue final state — grouped Q_{g,n} = 1 iff n in (cum[g-1], cum[g]]
    for g in gs:
        for n in ns:
            q_gn = 1 if (cum[g - 1] < n <= cum[g]) else 0
            m.addConstr(w[T, g, n] == q_gn)

    # (28) at most one group per queue position
    for t in range(0, T + 1):
        for n in ns:
            m.addConstr(gp.quicksum(w[t, g, n] for g in gs) <= 1)

    # (29) at most one retrieval per step
    for t in ts:
        m.addConstr(gp.quicksum(k[t, g, n] for g in gs for n in ns) <= 1)

    # (30) queue fills from position 1 upward
    for t in range(0, T + 1):
        for n in range(1, N):
            m.addConstr(
                gp.quicksum(w[t, g, n]     for g in gs) >=
                gp.quicksum(w[t, g, n + 1] for g in gs)
            )

    # (31) (32) queue accumulation
    for t in ts:
        for g in gs:
            m.addConstr(
                gp.quicksum(w[t, g, n] for n in ns) ==
                gp.quicksum(w[t - 1, g, n] + k[t, g, n] for n in ns)
            )
    for t in ts:
        for g in gs:
            for n in ns:
                m.addConstr(w[t, g, n] == w[t - 1, g, n] + k[t, g, n])

    # (33) link retrievals to bay occupancy
    for t in ts:
        for g in gs:
            m.addConstr(
                gp.quicksum(x[t - 1, g, s, h] for s in ss for h in hs) ==
                gp.quicksum(k[t, g, n] for n in ns) +
                gp.quicksum(x[t, g, s, h] for s in ss for h in hs)
            )

    # (34) duplicate of (10) — appears in the paper as a separate constraint
    for t in ts:
        for g in gs:
            for s in ss:
                for h in hs:
                    m.addConstr(
                        x[t, g, s, h] + z[t, g, s, h] >=
                        x[t - 1, g, s, h] + y[t, g, s, h]
                    )

    # (35) each retrieval corresponds to a z and a k
    for t in ts:
        for g in gs:
            m.addConstr(
                gp.quicksum(y[t, g, s, h] for s in ss for h in hs) +
                gp.quicksum(k[t, g, n]    for n in ns) ==
                gp.quicksum(z[t, g, s, h] for s in ss for h in hs)
            )

    # (36) first N steps are all retrieval steps
    for t in range(1, N + 1):
        m.addConstr(
            gp.quicksum(z[t, g, s, h] for g in gs for s in ss for h in hs) == 1
        )

    m.optimize()

    status = m.Status
    if m.SolCount == 0:
        return dict(obj=None, optimal=False, time_out=(status in (9, 11)),
                    n_vars=m.NumVars, n_constrs=m.NumConstrs, solve_time=m.Runtime)
    return dict(
        obj=int(round(m.ObjVal)), optimal=(status == 2),
        time_out=(status in (9, 11)),
        n_vars=m.NumVars, n_constrs=m.NumConstrs, solve_time=m.Runtime,
    )


# ================================================================ #
#  BRP m2 — unrestricted, grouped priorities                        #
# ================================================================ #

def _solve_brp_m2_grouped(
    yard: Yard,
    N: int, S: int, H: int, G: int,
    T_ub: int,
    time_limit_s: float,
    output_flag: int,
) -> Dict:
    """
    Unrestricted BRP m2 with grouped priorities.

    Constraint (48) (stack-order for unique priorities) is REPLACED by the
    aggregated forms of Constraints (49) and (50):

      (49-agg)  k[t,g,s,h] + sum_{l<g, i<h} x[t,l,s,i]       <= 1
      (50-agg)  k[t,g,s,h] + sum_{l<g, p!=s, all i} x[t,l,p,i] <= 1

    Together they enforce: group-g container at (s,h) can only be retrieved
    at step t if no container of a lower group exists anywhere in the bay.

    T = T_ub  (relocation steps only; retrievals are embedded in the time steps).
    """
    import gurobipy as gp
    from gurobipy import GRB

    T = T_ub
    cfg, _ = _yard_to_config_grouped(yard, S, H)

    m = gp.Model()
    m.Params.TimeLimit  = float(time_limit_s)
    m.Params.MIPGapAbs  = 0.99
    m.Params.OutputFlag = output_flag

    gs = range(1, G + 1)
    ss = range(1, S + 1)
    hs = range(1, H + 1)
    ts = range(1, T + 1)

    x = m.addVars(
        [(t, g, s, h) for t in range(0, T + 1) for g in gs for s in ss for h in hs],
        vtype=GRB.BINARY, name="x",
    )
    y = m.addVars(
        [(t, g, s, h) for t in ts for g in gs for s in ss for h in hs],
        vtype=GRB.BINARY, name="y",
    )
    z = m.addVars(
        [(t, g, s, h) for t in ts for g in gs for s in ss for h in hs],
        vtype=GRB.BINARY, name="z",
    )
    k = m.addVars(
        [(t, g, s, h) for t in ts for g in gs for s in ss for h in hs],
        vtype=GRB.BINARY, name="k",
    )

    m.setObjective(
        gp.quicksum(y[t, g, s, h] for t in ts for g in gs for s in ss for h in hs),
        GRB.MINIMIZE,
    )

    _build_common_constraints(m, x, y, z, cfg, gs, ss, hs, ts, T, H, G, N)

    # (41) bay empty at end
    m.addConstr(
        gp.quicksum(x[T, g, s, h] for g in gs for s in ss for h in hs) == 0
    )

    # (42) slot occupancy during retrieval
    for t in ts:
        for s in ss:
            for h in hs:
                m.addConstr(
                    gp.quicksum(x[t, g, s, h] + k[t, g, s, h] for g in gs) <= 1
                )

    # (43) bay state consistency with retrievals
    for t in ts:
        for g in gs:
            m.addConstr(
                gp.quicksum(x[t - 1, g, s, h] for s in ss for h in hs) ==
                gp.quicksum(x[t, g, s, h] + k[t, g, s, h] for s in ss for h in hs)
            )

    # (44) flow link
    for t in ts:
        for g in gs:
            for s in ss:
                for h in hs:
                    m.addConstr(
                        z[t, g, s, h] + k[t, g, s, h] ==
                        y[t, g, s, h] + x[t - 1, g, s, h] - x[t, g, s, h]
                    )

    # (46) retrieval group ordering
    for t in ts:
        for g in range(2, G + 1):
            m.addConstr(
                gp.quicksum(k[t, g, s, h] for s in ss for h in hs) <=
                gp.quicksum(
                    k[u, g - 1, s, h]
                    for u in range(1, t + 1) for s in ss for h in hs
                )
            )

    # (47) no two same-stack retrievals across adjacent tiers
    for t in ts:
        for g in range(1, G):
            for s in ss:
                for h in range(1, H):
                    m.addConstr(
                        k[t, g, s, h] +
                        gp.quicksum(k[t, l, s, h + 1] for l in range(g + 1, G + 1))
                        <= 1
                    )

    # (49-agg) no lower-group container below (s,h) in same stack
    for t in ts:
        for g in range(2, G + 1):
            for s in ss:
                for h in hs:
                    if h > 1:
                        m.addConstr(
                            k[t, g, s, h] +
                            gp.quicksum(
                                x[t, l, s, i]
                                for l in range(1, g) for i in range(1, h)
                            ) <= 1
                        )

    # (50-agg) no lower-group container in any other stack
    for t in ts:
        for g in range(2, G + 1):
            for s in ss:
                for h in hs:
                    other = [p for p in ss if p != s]
                    if not other:
                        continue
                    m.addConstr(
                        k[t, g, s, h] +
                        gp.quicksum(
                            x[t, l, p, i]
                            for l in range(1, g) for p in other for i in hs
                        ) <= 1
                    )

    m.optimize()

    status = m.Status
    if m.SolCount == 0:
        return dict(obj=None, optimal=False, time_out=(status in (9, 11)),
                    n_vars=m.NumVars, n_constrs=m.NumConstrs, solve_time=m.Runtime)
    return dict(
        obj=int(round(m.ObjVal)), optimal=(status == 2),
        time_out=(status in (9, 11)),
        n_vars=m.NumVars, n_constrs=m.NumConstrs, solve_time=m.Runtime,
    )


# ================================================================ #
#  r-BRP m1 — restricted, grouped priorities                        #
# ================================================================ #

def _solve_rbrp_m1_grouped(
    yard: Yard,
    N: int, S: int, H: int, G: int,
    group_sizes: Dict[int, int],
    T_ub: int,
    time_limit_s: float,
    output_flag: int,
) -> Dict:
    """
    Restricted r-BRP m1 with grouped priorities.

    Extends BRP m1 grouped with Constraint (51) adapted for groups:

      (51-grp)  sum_{j>g} sum_h z[t,j,s,h]
                  <= w[t-1, g, cum[g]]
                   + sum_{i<=g} sum_h x[t, i, s, h]

    Semantics: a container of group j > g can be relocated from stack s at
    step t only if all containers of group g have been retrieved to the queue
    (w[t-1,g,cum[g]] = 1, since cum[g] is the last queue position of group g
    and the queue fills in order) OR some group-i (i<=g) container is still
    present in stack s.
    """
    import gurobipy as gp
    from gurobipy import GRB

    T = T_ub + N
    cfg, _ = _yard_to_config_grouped(yard, S, H)
    cum = _cumulative_positions(G, group_sizes)

    m = gp.Model()
    m.Params.TimeLimit  = float(time_limit_s)
    m.Params.MIPGapAbs  = 0.99
    m.Params.OutputFlag = output_flag

    gs = range(1, G + 1)
    ss = range(1, S + 1)
    hs = range(1, H + 1)
    ts = range(1, T + 1)
    ns = range(1, N + 1)

    x = m.addVars(
        [(t, g, s, h) for t in range(0, T + 1) for g in gs for s in ss for h in hs],
        vtype=GRB.BINARY, name="x",
    )
    y = m.addVars(
        [(t, g, s, h) for t in ts for g in gs for s in ss for h in hs],
        vtype=GRB.BINARY, name="y",
    )
    z = m.addVars(
        [(t, g, s, h) for t in ts for g in gs for s in ss for h in hs],
        vtype=GRB.BINARY, name="z",
    )
    k = m.addVars(
        [(t, g, n) for t in ts for g in gs for n in ns],
        vtype=GRB.BINARY, name="k",
    )
    w = m.addVars(
        [(t, g, n) for t in range(0, T + 1) for g in gs for n in ns],
        vtype=GRB.BINARY, name="w",
    )

    m.setObjective(
        gp.quicksum(y[t, g, s, h] for t in ts for g in gs for s in ss for h in hs),
        GRB.MINIMIZE,
    )

    _build_common_constraints(m, x, y, z, cfg, gs, ss, hs, ts, T, H, G, N)

    for t in ts:
        for g in gs:
            for s in ss:
                for h in hs:
                    m.addConstr(
                        x[t, g, s, h] + z[t, g, s, h] >=
                        x[t - 1, g, s, h] + y[t, g, s, h]
                    )

    for t in range(2, T + 1):
        m.addConstr(
            gp.quicksum(z[t - 1, g, s, h] for g in gs for s in ss for h in hs) >=
            gp.quicksum(z[t,     g, s, h] for g in gs for s in ss for h in hs)
        )

    m.addConstr(
        gp.quicksum(x[T, g, s, h] for g in gs for s in ss for h in hs) == 0
    )

    for g in gs:
        for n in ns:
            m.addConstr(w[0, g, n] == 0)

    for g in gs:
        for n in ns:
            q_gn = 1 if (cum[g - 1] < n <= cum[g]) else 0
            m.addConstr(w[T, g, n] == q_gn)

    for t in range(0, T + 1):
        for n in ns:
            m.addConstr(gp.quicksum(w[t, g, n] for g in gs) <= 1)

    for t in ts:
        m.addConstr(gp.quicksum(k[t, g, n] for g in gs for n in ns) <= 1)

    for t in range(0, T + 1):
        for n in range(1, N):
            m.addConstr(
                gp.quicksum(w[t, g, n]     for g in gs) >=
                gp.quicksum(w[t, g, n + 1] for g in gs)
            )

    for t in ts:
        for g in gs:
            m.addConstr(
                gp.quicksum(w[t, g, n] for n in ns) ==
                gp.quicksum(w[t - 1, g, n] + k[t, g, n] for n in ns)
            )
    for t in ts:
        for g in gs:
            for n in ns:
                m.addConstr(w[t, g, n] == w[t - 1, g, n] + k[t, g, n])

    for t in ts:
        for g in gs:
            m.addConstr(
                gp.quicksum(x[t - 1, g, s, h] for s in ss for h in hs) ==
                gp.quicksum(k[t, g, n] for n in ns) +
                gp.quicksum(x[t, g, s, h] for s in ss for h in hs)
            )

    for t in ts:
        for g in gs:
            m.addConstr(
                gp.quicksum(y[t, g, s, h] for s in ss for h in hs) +
                gp.quicksum(k[t, g, n]    for n in ns) ==
                gp.quicksum(z[t, g, s, h] for s in ss for h in hs)
            )

    for t in range(1, N + 1):
        m.addConstr(
            gp.quicksum(z[t, g, s, h] for g in gs for s in ss for h in hs) == 1
        )

    # (51-grp) restricted relocation: adapted Constraint (51) for grouped priorities.
    # w[t-1, g, cum[g]] = 1 iff all n_g containers of group g have been retrieved
    # (queue position cum[g] is the last position of group g; queue fills in order).
    for t in ts:
        for g in range(1, G):
            for s in ss:
                m.addConstr(
                    gp.quicksum(
                        z[t, j, s, h] for j in range(g + 1, G + 1) for h in hs
                    ) <=
                    w[t - 1, g, cum[g]] +
                    gp.quicksum(x[t, i, s, h] for i in range(1, g + 1) for h in hs)
                )

    m.optimize()

    status = m.Status
    if m.SolCount == 0:
        return dict(obj=None, optimal=False, time_out=(status in (9, 11)),
                    n_vars=m.NumVars, n_constrs=m.NumConstrs, solve_time=m.Runtime)
    return dict(
        obj=int(round(m.ObjVal)), optimal=(status == 2),
        time_out=(status in (9, 11)),
        n_vars=m.NumVars, n_constrs=m.NumConstrs, solve_time=m.Runtime,
    )


# ================================================================ #
#  r-BRP m2 — restricted, grouped priorities                        #
# ================================================================ #

def _solve_rbrp_m2_grouped(
    yard: Yard,
    N: int, S: int, H: int, G: int,
    T_ub: int,
    time_limit_s: float,
    output_flag: int,
) -> Dict:
    """
    Restricted r-BRP m2 with grouped priorities.

    Combines:
      - BRP m2 grouped [(49-agg)+(50-agg) replace (48)]
      - Constraint (52) for restricted relocation:

          (52)  sum_{j>g} sum_h z[t,j,s,h]
                  <= sum_{u<=t} sum_{r,h} k[u,g,r,h]
                   + sum_{i<=g} sum_h x[t,i,s,h]

    T = T_ub  (relocation steps only).
    """
    import gurobipy as gp
    from gurobipy import GRB

    T = T_ub
    cfg, _ = _yard_to_config_grouped(yard, S, H)

    m = gp.Model()
    m.Params.TimeLimit  = float(time_limit_s)
    m.Params.MIPGapAbs  = 0.99
    m.Params.OutputFlag = output_flag

    gs = range(1, G + 1)
    ss = range(1, S + 1)
    hs = range(1, H + 1)
    ts = range(1, T + 1)

    x = m.addVars(
        [(t, g, s, h) for t in range(0, T + 1) for g in gs for s in ss for h in hs],
        vtype=GRB.BINARY, name="x",
    )
    y = m.addVars(
        [(t, g, s, h) for t in ts for g in gs for s in ss for h in hs],
        vtype=GRB.BINARY, name="y",
    )
    z = m.addVars(
        [(t, g, s, h) for t in ts for g in gs for s in ss for h in hs],
        vtype=GRB.BINARY, name="z",
    )
    k = m.addVars(
        [(t, g, s, h) for t in ts for g in gs for s in ss for h in hs],
        vtype=GRB.BINARY, name="k",
    )

    m.setObjective(
        gp.quicksum(y[t, g, s, h] for t in ts for g in gs for s in ss for h in hs),
        GRB.MINIMIZE,
    )

    _build_common_constraints(m, x, y, z, cfg, gs, ss, hs, ts, T, H, G, N)

    # (41) bay empty at end
    m.addConstr(
        gp.quicksum(x[T, g, s, h] for g in gs for s in ss for h in hs) == 0
    )

    # (42) slot occupancy consistency
    for t in ts:
        for s in ss:
            for h in hs:
                m.addConstr(
                    gp.quicksum(x[t, g, s, h] + k[t, g, s, h] for g in gs) <= 1
                )

    # (43) bay state consistency
    for t in ts:
        for g in gs:
            m.addConstr(
                gp.quicksum(x[t - 1, g, s, h] for s in ss for h in hs) ==
                gp.quicksum(x[t, g, s, h] + k[t, g, s, h] for s in ss for h in hs)
            )

    # (44) flow link
    for t in ts:
        for g in gs:
            for s in ss:
                for h in hs:
                    m.addConstr(
                        z[t, g, s, h] + k[t, g, s, h] ==
                        y[t, g, s, h] + x[t - 1, g, s, h] - x[t, g, s, h]
                    )

    # (46) retrieval group ordering
    for t in ts:
        for g in range(2, G + 1):
            m.addConstr(
                gp.quicksum(k[t, g, s, h] for s in ss for h in hs) <=
                gp.quicksum(
                    k[u, g - 1, s, h]
                    for u in range(1, t + 1) for s in ss for h in hs
                )
            )

    # (47) no two same-stack retrievals across adjacent tiers
    for t in ts:
        for g in range(1, G):
            for s in ss:
                for h in range(1, H):
                    m.addConstr(
                        k[t, g, s, h] +
                        gp.quicksum(k[t, l, s, h + 1] for l in range(g + 1, G + 1))
                        <= 1
                    )

    # (49-agg) no lower-group container below (s,h) in same stack
    for t in ts:
        for g in range(2, G + 1):
            for s in ss:
                for h in hs:
                    if h > 1:
                        m.addConstr(
                            k[t, g, s, h] +
                            gp.quicksum(
                                x[t, l, s, i]
                                for l in range(1, g) for i in range(1, h)
                            ) <= 1
                        )

    # (50-agg) no lower-group container in any other stack
    for t in ts:
        for g in range(2, G + 1):
            for s in ss:
                for h in hs:
                    other = [p for p in ss if p != s]
                    if not other:
                        continue
                    m.addConstr(
                        k[t, g, s, h] +
                        gp.quicksum(
                            x[t, l, p, i]
                            for l in range(1, g) for p in other for i in hs
                        ) <= 1
                    )

    # (52) restricted relocation for m2 grouped
    for t in ts:
        for g in range(1, G):
            for s in ss:
                m.addConstr(
                    gp.quicksum(
                        z[t, j, s, h] for j in range(g + 1, G + 1) for h in hs
                    ) <=
                    gp.quicksum(
                        k[u, g, r, h]
                        for u in range(1, t + 1) for r in ss for h in hs
                    ) +
                    gp.quicksum(x[t, i, s, h] for i in range(1, g + 1) for h in hs)
                )

    m.optimize()

    status = m.Status
    if m.SolCount == 0:
        return dict(obj=None, optimal=False, time_out=(status in (9, 11)),
                    n_vars=m.NumVars, n_constrs=m.NumConstrs, solve_time=m.Runtime)
    return dict(
        obj=int(round(m.ObjVal)), optimal=(status == 2),
        time_out=(status in (9, 11)),
        n_vars=m.NumVars, n_constrs=m.NumConstrs, solve_time=m.Runtime,
    )


# ================================================================ #
#  BaseAlgorithm — Unrestricted CRP-D                               #
# ================================================================ #

class DeMeloSilva2018BRPGrouped(BaseAlgorithm):
    """
    de Melo da Silva et al. (2018) Unrestricted BRP — CRP-D grouped priorities.

    Variant m1 (default): T = UB + N; stronger LP relaxation via queue variables
    with grouped Q_gn.
    Variant m2: T = UB (relocations only); Constraints (49)+(50) enforce grouped
    retrieval precedence.

    Requires: Gurobi with Named-User Academic License.
    """

    name                = "de Melo da Silva (2018) BRP Grouped [Gurobi]"
    category            = "Exact"
    requires_solver     = True
    solver_backend      = "gurobi"
    description         = (
        "Time-indexed MIP for Unrestricted BRP with grouped/duplicate priorities "
        "(CRP-D).  de Melo da Silva, Toulouse & Wolfler Calvo — EJOR 271 (2018). "
        "m1: T=UB+N, grouped Q_gn (Eq. 27);  "
        "m2: T=UB, Constraints (49)+(50) replace (48). "
        "[Requires Gurobi license]"
    )
    compatible_problems = ["CRP-D"]
    geometry            = "single-bay"
    objectives          = ["relocations"]
    fidelity            = "faithful"
    def __init__(self, config: Optional[AlgorithmConfig] = None) -> None:
        super().__init__(config)

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        cfg   = self.config
        extra = cfg.extra or {}

        variant      = str(extra.get("variant",      "m1"))
        time_limit_s = float(extra.get("time_limit_s", 3600.0))
        output_flag  = int(extra.get("output_flag",   0))
        n_seeds = 1  # multi-seed eval removed; single run only
        all_metrics: List[Dict] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.reset(seed=seed, options={"skip_auto_retrieve": True})

            yard = env.yard
            N    = len(env.containers)
            S    = env.config.num_bays * env.config.num_rows
            H    = env.config.max_tiers
            G    = len(set(c.priority for c in env.containers))

            _, group_sizes = _yard_to_config_grouped(yard, S, H)
            T_ub = max(1, _greedy_ub_grouped(yard, G, S, H))

            try:
                import gurobipy  # noqa: F401
            except ImportError:
                print(
                    "[DeMeloSilva2018BRPGrouped] ERROR: gurobipy not installed.",
                    file=sys.stderr, flush=True,
                )
                self._push(
                    result_queue, step=seed + 1, metric=float("inf"),
                    metrics={"relocations": float("inf"), "error": 1.0},
                    progress=(seed + 1) / n_seeds,
                )
                continue

            t0 = time.perf_counter()

            if variant == "m2":
                result = _solve_brp_m2_grouped(
                    yard, N, S, H, G, T_ub, time_limit_s, output_flag
                )
            else:
                result = _solve_brp_m1_grouped(
                    yard, N, S, H, G, group_sizes, T_ub, time_limit_s, output_flag
                )

            elapsed = time.perf_counter() - t0
            n_reloc = result["obj"] if result["obj"] is not None else -1
            primary = float(max(n_reloc, 0)) if result["obj"] is not None else float("inf")

            if primary < self._best_metric:
                self._best_metric   = primary
                self._best_solution = []

            metrics: Dict = {
                "relocations":    float(max(n_reloc, 0)),
                "steps":          float(max(n_reloc, 0)),
                "optimal_proven": 1.0 if result["optimal"] else 0.0,
                "time_out":       1.0 if result["time_out"] else 0.0,
                "T_ub":           float(T_ub),
                "G":              float(G),
                "N":              float(N),
                "n_vars":         float(result["n_vars"]),
                "n_constrs":      float(result["n_constrs"]),
                "solve_time_s":   round(elapsed, 4),
                "feasible":       0.0 if result["obj"] is None else 1.0,
            }
            all_metrics.append(metrics)

            self._push(
                result_queue,
                step     = seed + 1,
                metric   = primary,
                metrics  = metrics,
                progress = (seed + 1) / n_seeds,
                snapshot = env.get_state_snapshot(),
            )
            print(
                f"[DeMeloSilva2018BRPGrouped/{variant}] seed={seed}  "
                f"G={G} N={N} S={S} H={H} T_ub={T_ub}  "
                f"reloc={n_reloc}  opt={result['optimal']}  t={elapsed:.2f}s",
                file=sys.stderr, flush=True,
            )

        if all_metrics:
            agg = {
                k: float(np.mean([m[k] for m in all_metrics if k in m]))
                for k in all_metrics[0]
            }
            self._push(result_queue, step=n_seeds, metric=self._best_metric,
                       metrics=agg, progress=1.0)

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "variant": {
                "type": "str", "default": "m1", "options": ["m1", "m2"],
                "label": "Model variant",
                "help": (
                    "m1: T=UB+N; stronger LP relaxation via grouped queue variables. "
                    "m2: T=UB only; more compact using Constraints (49)+(50)."
                ),
            },
            "time_limit_s": {
                "type": "float", "default": 3600.0, "min": 1.0, "max": 86400.0,
                "label": "Time limit (s)",
            },
        })
        return base


# ================================================================ #
#  BaseAlgorithm — Restricted CRP-D                                 #
# ================================================================ #

class DeMeloSilva2018RBRPGrouped(BaseAlgorithm):
    """
    de Melo da Silva et al. (2018) Restricted r-BRP — CRP-D grouped priorities.

    Variant m1 (default): T = UB + N; Constraint (51) adapted for groups
    (uses w[t-1,g,cum[g]] to test whether ALL containers of group g were retrieved).
    Variant m2: T = UB; Constraint (52) enforces restricted relocation.

    Use with CRP-D problem config  extra["restricted_relocation"] = true
    to activate the restricted move semantics in the environment.

    Requires: Gurobi with Named-User Academic License.
    """

    name                = "de Melo da Silva (2018) r-BRP Grouped [Gurobi]"
    category            = "Exact"
    requires_solver     = True
    solver_backend      = "gurobi"
    description         = (
        "Time-indexed MIP for Restricted r-BRP with grouped/duplicate priorities "
        "(CRP-D).  de Melo da Silva, Toulouse & Wolfler Calvo — EJOR 271 (2018). "
        "m1: T=UB+N; adapted Constraint (51) for groups;  "
        "m2: T=UB; Constraint (52) for restricted relocation. "
        "[Requires Gurobi license]"
    )
    compatible_problems = ["CRP-D"]
    geometry            = "single-bay"
    objectives          = ["relocations"]
    fidelity            = "faithful"
    def __init__(self, config: Optional[AlgorithmConfig] = None) -> None:
        super().__init__(config)

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        cfg   = self.config
        extra = cfg.extra or {}

        variant      = str(extra.get("variant",      "m1"))
        time_limit_s = float(extra.get("time_limit_s", 3600.0))
        output_flag  = int(extra.get("output_flag",   0))
        n_seeds = 1  # multi-seed eval removed; single run only
        all_metrics: List[Dict] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.reset(seed=seed, options={"skip_auto_retrieve": True})

            yard = env.yard
            N    = len(env.containers)
            S    = env.config.num_bays * env.config.num_rows
            H    = env.config.max_tiers
            G    = len(set(c.priority for c in env.containers))

            _, group_sizes = _yard_to_config_grouped(yard, S, H)
            T_ub = max(1, _greedy_ub_grouped(yard, G, S, H))

            try:
                import gurobipy  # noqa: F401
            except ImportError:
                print(
                    "[DeMeloSilva2018RBRPGrouped] ERROR: gurobipy not installed.",
                    file=sys.stderr, flush=True,
                )
                self._push(
                    result_queue, step=seed + 1, metric=float("inf"),
                    metrics={"relocations": float("inf"), "error": 1.0},
                    progress=(seed + 1) / n_seeds,
                )
                continue

            t0 = time.perf_counter()

            if variant == "m2":
                result = _solve_rbrp_m2_grouped(
                    yard, N, S, H, G, T_ub, time_limit_s, output_flag
                )
            else:
                result = _solve_rbrp_m1_grouped(
                    yard, N, S, H, G, group_sizes, T_ub, time_limit_s, output_flag
                )

            elapsed = time.perf_counter() - t0
            n_reloc = result["obj"] if result["obj"] is not None else -1
            primary = float(max(n_reloc, 0)) if result["obj"] is not None else float("inf")

            if primary < self._best_metric:
                self._best_metric   = primary
                self._best_solution = []

            metrics: Dict = {
                "relocations":    float(max(n_reloc, 0)),
                "steps":          float(max(n_reloc, 0)),
                "optimal_proven": 1.0 if result["optimal"] else 0.0,
                "time_out":       1.0 if result["time_out"] else 0.0,
                "T_ub":           float(T_ub),
                "G":              float(G),
                "N":              float(N),
                "n_vars":         float(result["n_vars"]),
                "n_constrs":      float(result["n_constrs"]),
                "solve_time_s":   round(elapsed, 4),
                "feasible":       0.0 if result["obj"] is None else 1.0,
            }
            all_metrics.append(metrics)

            self._push(
                result_queue,
                step     = seed + 1,
                metric   = primary,
                metrics  = metrics,
                progress = (seed + 1) / n_seeds,
                snapshot = env.get_state_snapshot(),
            )
            print(
                f"[DeMeloSilva2018RBRPGrouped/{variant}] seed={seed}  "
                f"G={G} N={N} S={S} H={H} T_ub={T_ub}  "
                f"reloc={n_reloc}  opt={result['optimal']}  t={elapsed:.2f}s",
                file=sys.stderr, flush=True,
            )

        if all_metrics:
            agg = {
                k: float(np.mean([m[k] for m in all_metrics if k in m]))
                for k in all_metrics[0]
            }
            self._push(result_queue, step=n_seeds, metric=self._best_metric,
                       metrics=agg, progress=1.0)

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "variant": {
                "type": "str", "default": "m1", "options": ["m1", "m2"],
                "label": "Model variant",
                "help": (
                    "m1: T=UB+N; adapted Constraint (51) for grouped priorities. "
                    "m2: T=UB; Constraint (52) for restricted relocation."
                ),
            },
            "time_limit_s": {
                "type": "float", "default": 3600.0, "min": 1.0, "max": 86400.0,
                "label": "Time limit (s)",
            },
        })
        return base
