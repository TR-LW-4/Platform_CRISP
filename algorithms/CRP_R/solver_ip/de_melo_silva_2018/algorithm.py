"""
de Melo da Silva, Toulouse, Wolfler Calvo (2018)
"A new effective unified model for solving the Pre-marshalling and
Block Relocation Problems"
European Journal of Operational Research 271 (2018) 40–56.

This module implements the **Restricted BRP** (r-BRP) formulations only:
  m1 (r-BRP m1): Eq. (23)–(36) + Constraint (51).  Stronger LP relaxation.
  m2 (r-BRP m2): Eq. (39)–(50) + Constraint (52).  More compact, often faster.

Select with  extra["variant"] = "m1"  (default) or  "m2".

Solver: Gurobi (gurobipy).  Academic Named-User License recommended.

Variable glossary
-----------------
x[t,g,s,h]  = 1 if a container of group g occupies slot (s,h) at end of step t
y[t,g,s,h]  = 1 if a container of group g enters slot (s,h) during step t
z[t,g,s,h]  = 1 if a container of group g leaves slot (s,h) during step t
                (= pick-up for relocation; also retrieval in m1)
k_m1[t,g,n] = 1 if group g is retrieved to queue position n during step t  (m1)
w_m1[t,g,n] = 1 if group g occupies queue position n at end of step t      (m1)
k_m2[t,g,s,h] = 1 if group g is removed (retrieved) from slot (s,h) at t  (m2)

All variables are 0/1 binary and 1-indexed in the MIP.

Upper bound T
-------------
T = r-BRP upper bound obtained by running the Tanaka B&B with time_limit_ub_s.
If Tanaka is unavailable we fall back to the MinMax heuristic (Caserta 2012).
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
                    solve_time=m.Runtime)
    obj = int(round(m.ObjVal))
    return dict(obj=obj, optimal=(status == 2),
                time_out=(status in (9,11)),
                n_vars=m.NumVars, n_constrs=m.NumConstrs,
                solve_time=m.Runtime)


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
                    solve_time=m.Runtime)
    obj = int(round(m.ObjVal))
    return dict(obj=obj, optimal=(status == 2),
                time_out=(status in (9,11)),
                n_vars=m.NumVars, n_constrs=m.NumConstrs,
                solve_time=m.Runtime)


# ================================================================ #
#  BaseAlgorithm subclass — CRP-R                                    #
# ================================================================ #

class DeMeloSilva2018RBRP(BaseAlgorithm):
    """
    de Melo da Silva et al. (2018) Restricted BRP — time-indexed MIP.

    Variant m1 (default): stronger LP relaxation, solves ≈82 % of Caserta
    benchmark classes data3-3 … data5-6 to optimality within 1 h (CPLEX).

    Variant m2: more compact formulation; weaker LP but often faster in
    practice for instances where a feasible solution is hard to find.

    Requires: Gurobi with Named-User Academic License.
    """

    name                = "de Melo da Silva (2018) r-BRP [Gurobi]"
    category            = "Exact"
    requires_solver     = True
    solver_backend      = "gurobi"
    description         = (
        "Time-indexed MIP for Restricted BRP (r-BRP m1 / m2). "
        "de Melo da Silva, Toulouse & Wolfler Calvo — EJOR 271 (2018). "
        "variant=m1: stronger LP; variant=m2: more compact. "
        "[Requires Gurobi license]"
    )
    compatible_problems = ["CRP-R"]
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

            yard  = env.yard
            C     = len(env.containers)
            S     = env.config.num_bays * env.config.num_rows
            H     = env.config.max_tiers
            G     = C  # unique priorities → G = N

            # Upper bound T via MinMax heuristic (Caserta 2012)
            T_ub = _minmax_upper_bound(yard, C, S, H)
            if T_ub <= 0:
                T_ub = 1

            t0 = time.perf_counter()

            try:
                import gurobipy  # noqa: F401
            except ImportError:
                print("[DeMeloSilva2018RBRP] ERROR: gurobipy not installed.",
                      file=sys.stderr, flush=True)
                self._push(result_queue, step=seed+1, metric=float("inf"),
                           metrics={"relocations": float("inf"), "error": 1.0},
                           progress=(seed+1)/n_seeds)
                continue

            if variant == "m2":
                result = _solve_rbrp_m2(yard, C, S, H, G, T_ub,
                                        time_limit_s, output_flag)
            else:
                result = _solve_rbrp_m1(yard, C, S, H, G, T_ub,
                                        time_limit_s, output_flag)

            elapsed   = time.perf_counter() - t0
            n_reloc   = result["obj"] if result["obj"] is not None else -1
            primary   = float(max(n_reloc, 0)) if result["obj"] is not None else float("inf")

            if primary < self._best_metric:
                self._best_metric   = primary
                self._best_solution = []

            metrics: Dict = {
                "relocations":    float(max(n_reloc, 0)),
                "steps":          float(max(n_reloc, 0)),
                "time":           float(max(n_reloc, 0)),
                "optimal_proven": 1.0 if result["optimal"] else 0.0,
                "time_out":       1.0 if result["time_out"] else 0.0,
                "T_ub":           float(T_ub),
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
                f"[DeMeloSilva2018RBRP/{variant}] seed={seed}  "
                f"C={C} S={S} H={H} T_ub={T_ub}  "
                f"reloc={n_reloc}  opt={result['optimal']}  "
                f"t={elapsed:.2f}s",
                file=sys.stderr, flush=True,
            )

        if all_metrics:
            agg = {k: float(np.mean([m[k] for m in all_metrics if k in m]))
                   for k in all_metrics[0]}
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
                    "m1: stronger LP relaxation (≈82% instances solved, Caserta bench). "
                    "m2: more compact (faster per node, weaker LP, ≈66% instances solved)."
                ),
            },
            "time_limit_s": {
                "type": "float", "default": 3600.0, "min": 1.0, "max": 86400.0,
                "label": "Time limit (s)",
            },
        })
        return base
