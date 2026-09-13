"""
DeMeloSilva2018BRP
<2018> <exact> <unrestricted> <single-bay> <CRP-U>
Time-indexed MIP for unrestricted BRP (m1 / m2)
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
#  Helpers                                                            #
# ================================================================ #

def _yard_to_config(yard: Yard, S: int, H: int, G: int) -> Dict[Tuple[int,int,int], int]:
    """Return {(g, s, h): 1} for each occupied slot (1-indexed)."""
    config: Dict[Tuple[int,int,int], int] = {}
    stack_list = sorted(yard.stacks.values(), key=lambda st: (st.bay, st.row))
    for s_idx, stack in enumerate(stack_list, 1):
        for h_idx, cont in enumerate(stack.containers, 1):
            config[(cont.priority, s_idx, h_idx)] = 1
    return config


def _minmax_ub(yard: Yard, C: int, S: int, H: int) -> int:
    """Caserta 2012 MinMax heuristic — upper bound on relocations."""
    stacks = [
        [c.priority for c in st.containers]
        for st in sorted(yard.stacks.values(), key=lambda st: (st.bay, st.row))
    ]
    height = [len(s) for s in stacks]

    def min_p(si: int) -> int:
        return min(stacks[si]) if stacks[si] else C + 1

    mp_ = [min_p(s) for s in range(S)]
    n_reloc = 0

    for tp in range(1, C + 1):
        ts = ti = None
        for si in range(S):
            for i, p in enumerate(stacks[si]):
                if p == tp:
                    ts, ti = si, i
                    break
            if ts is not None:
                break
        if ts is None:
            continue
        while len(stacks[ts]) > ti + 1:
            n_reloc += 1
            r = stacks[ts][-1]
            good = [s for s in range(S) if s != ts and height[s] < H and mp_[s] > r]
            if good:
                dst = min(good, key=lambda s: mp_[s])
            else:
                avail = [s for s in range(S) if s != ts and height[s] < H]
                if not avail:
                    break
                dst = max(avail, key=lambda s: mp_[s])
            stacks[ts].pop()
            height[ts] -= 1
            stacks[dst].append(r)
            height[dst] += 1
            mp_[dst] = min_p(dst)
            mp_[ts]  = min_p(ts)
        stacks[ts] = [p for p in stacks[ts] if p != tp]
        height[ts] -= 1
        mp_[ts] = min_p(ts)

    return n_reloc


# ================================================================ #
#  Shared model builder                                               #
# ================================================================ #

def _build_common_constraints(m, x, y, z, cfg, gs, ss, hs, ts, T, H, G, C):
    """Build constraints shared by BRP m1 and m2 (Eq 2,4,8,9,15-17,20)."""
    import gurobipy as gp

    # (2) initialise
    for g in gs:
        for s in ss:
            for h in hs:
                m.addConstr(x[0,g,s,h] == cfg.get((g,s,h), 0))

    # (4) no gaps
    for t in range(0, T+1):
        for s in ss:
            for h in range(1, H):
                m.addConstr(
                    gp.quicksum(x[t,g,s,h] for g in gs) >=
                    gp.quicksum(x[t,g,s,h+1] for g in gs)
                )

    # (8) at most one y per step
    for t in ts:
        m.addConstr(gp.quicksum(y[t,g,s,h] for g in gs for s in ss for h in hs) <= 1)

    # (9) at most one z per step
    for t in ts:
        m.addConstr(gp.quicksum(z[t,g,s,h] for g in gs for s in ss for h in hs) <= 1)

    # (15)-(17) LIFO
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

    # (20) no pick-up and drop-off same stack
    for t in ts:
        for s in ss:
            m.addConstr(
                gp.quicksum(z[t,g,s,h] + y[t,g,s,h] for g in gs for h in hs) <= 1
            )


# ================================================================ #
#  BRP m1 solver (unrestricted)                                       #
# ================================================================ #

def _solve_brp_m1(
    yard: Yard,
    C: int, S: int, H: int, G: int,
    T_ub: int,
    time_limit_s: float,
    output_flag: int,
) -> Dict:
    """
    Unrestricted BRP m1.  T = T_ub + N (relocations + retrievals).
    """
    import gurobipy as gp
    from gurobipy import GRB

    N = C
    T = T_ub + N  # total time steps

    cfg = _yard_to_config(yard, S, H, G)

    m = gp.Model()
    m.Params.TimeLimit  = float(time_limit_s)
    m.Params.MIPGapAbs  = 0.99
    m.Params.OutputFlag = output_flag

    gs = range(1, G + 1)
    ss = range(1, S + 1)
    hs = range(1, H + 1)
    ts = range(1, T + 1)
    ns = range(1, N + 1)

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

    m.setObjective(
        gp.quicksum(y[t,g,s,h] for t in ts for g in gs for s in ss for h in hs),
        GRB.MINIMIZE,
    )

    _build_common_constraints(m, x, y, z, cfg, gs, ss, hs, ts, T, H, G, C)

    # (10) flow balance
    for t in ts:
        for g in gs:
            for s in ss:
                for h in hs:
                    m.addConstr(x[t,g,s,h] + z[t,g,s,h] >= x[t-1,g,s,h] + y[t,g,s,h])

    # (11) idle steps at end
    for t in range(2, T+1):
        m.addConstr(
            gp.quicksum(z[t-1,g,s,h] for g in gs for s in ss for h in hs) >=
            gp.quicksum(z[t,g,s,h]   for g in gs for s in ss for h in hs)
        )

    # (25) bay empty at end
    m.addConstr(gp.quicksum(x[T,g,s,h] for g in gs for s in ss for h in hs) == 0)

    # (26) queue init
    for g in gs:
        for n in ns:
            m.addConstr(w[0,g,n] == 0)

    # (27) queue final: unique priorities → Q_{g,g}=1
    for g in gs:
        for n in ns:
            m.addConstr(w[T,g,n] == (1 if g == n else 0))

    # (28)
    for t in range(0, T+1):
        for n in ns:
            m.addConstr(gp.quicksum(w[t,g,n] for g in gs) <= 1)

    # (29) at most one retrieval per step
    for t in ts:
        m.addConstr(gp.quicksum(k[t,g,n] for g in gs for n in ns) <= 1)

    # (30) queue order
    for t in range(0, T+1):
        for n in range(1, N):
            m.addConstr(
                gp.quicksum(w[t,g,n] for g in gs) >=
                gp.quicksum(w[t,g,n+1] for g in gs)
            )

    # (31) (32)
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

    # (33)
    for t in ts:
        for g in gs:
            m.addConstr(
                gp.quicksum(x[t-1,g,s,h] for s in ss for h in hs) ==
                gp.quicksum(k[t,g,n] for n in ns) +
                gp.quicksum(x[t,g,s,h] for s in ss for h in hs)
            )

    # (34)
    for t in ts:
        for g in gs:
            for s in ss:
                for h in hs:
                    m.addConstr(x[t,g,s,h] + z[t,g,s,h] >= x[t-1,g,s,h] + y[t,g,s,h])

    # (35)
    for t in ts:
        for g in gs:
            m.addConstr(
                gp.quicksum(y[t,g,s,h] for s in ss for h in hs) +
                gp.quicksum(k[t,g,n] for n in ns) ==
                gp.quicksum(z[t,g,s,h] for s in ss for h in hs)
            )

    # (36) first N steps each have z=1
    for t in range(1, N + 1):
        m.addConstr(
            gp.quicksum(z[t,g,s,h] for g in gs for s in ss for h in hs) == 1
        )

    m.optimize()

    status = m.Status
    if m.SolCount == 0:
        return dict(obj=None, optimal=False, time_out=(status in (9,11)),
                    n_vars=m.NumVars, n_constrs=m.NumConstrs, solve_time=m.Runtime)
    return dict(obj=int(round(m.ObjVal)), optimal=(status == 2),
                time_out=(status in (9,11)),
                n_vars=m.NumVars, n_constrs=m.NumConstrs, solve_time=m.Runtime)


# ================================================================ #
#  BRP m2 solver (unrestricted)                                       #
# ================================================================ #

def _solve_brp_m2(
    yard: Yard,
    C: int, S: int, H: int, G: int,
    T_ub: int,
    time_limit_s: float,
    output_flag: int,
) -> Dict:
    """
    Unrestricted BRP m2.  T = T_ub (relocations only; retrievals inline).
    """
    import gurobipy as gp
    from gurobipy import GRB

    T = T_ub  # only relocation steps needed
    cfg = _yard_to_config(yard, S, H, G)

    m = gp.Model()
    m.Params.TimeLimit  = float(time_limit_s)
    m.Params.MIPGapAbs  = 0.99
    m.Params.OutputFlag = output_flag

    gs = range(1, G + 1)
    ss = range(1, S + 1)
    hs = range(1, H + 1)
    ts = range(1, T + 1)

    x = m.addVars([(t,g,s,h) for t in range(0,T+1) for g in gs for s in ss for h in hs],
                  vtype=GRB.BINARY, name="x")
    y = m.addVars([(t,g,s,h) for t in ts for g in gs for s in ss for h in hs],
                  vtype=GRB.BINARY, name="y")
    z = m.addVars([(t,g,s,h) for t in ts for g in gs for s in ss for h in hs],
                  vtype=GRB.BINARY, name="z")
    k = m.addVars([(t,g,s,h) for t in ts for g in gs for s in ss for h in hs],
                  vtype=GRB.BINARY, name="k")

    m.setObjective(
        gp.quicksum(y[t,g,s,h] for t in ts for g in gs for s in ss for h in hs),
        GRB.MINIMIZE,
    )

    _build_common_constraints(m, x, y, z, cfg, gs, ss, hs, ts, T, H, G, C)

    # (41) bay empty at end
    m.addConstr(gp.quicksum(x[T,g,s,h] for g in gs for s in ss for h in hs) == 0)

    # (42)
    for t in ts:
        for s in ss:
            for h in hs:
                m.addConstr(gp.quicksum(x[t,g,s,h] + k[t,g,s,h] for g in gs) <= 1)

    # (43)
    for t in ts:
        for g in gs:
            m.addConstr(
                gp.quicksum(x[t-1,g,s,h] for s in ss for h in hs) ==
                gp.quicksum(x[t,g,s,h] + k[t,g,s,h] for s in ss for h in hs)
            )

    # (44)
    for t in ts:
        for g in gs:
            for s in ss:
                for h in hs:
                    m.addConstr(
                        z[t,g,s,h] + k[t,g,s,h] ==
                        y[t,g,s,h] + x[t-1,g,s,h] - x[t,g,s,h]
                    )

    # (46) retrieval order
    for t in ts:
        for g in range(2, G + 1):
            m.addConstr(
                gp.quicksum(k[t,g,s,h] for s in ss for h in hs) <=
                gp.quicksum(k[u,g-1,s,h] for u in range(1,t+1) for s in ss for h in hs)
            )

    # (47) no two same-stack retrievals at same tier
    for t in ts:
        for g in range(1, G):
            for s in ss:
                for h in range(1, H):
                    m.addConstr(
                        k[t,g,s,h] +
                        gp.quicksum(k[t,l,s,h+1] for l in range(g+1, G+1)) <= 1
                    )

    # (48) retrieval order within stack
    for t in ts:
        for g in range(2, G + 1):
            for s in ss:
                for h in range(2, H + 1):
                    m.addConstr(
                        k[t,g,s,h] <=
                        1 - gp.quicksum(x[t,l,s,h-1] for l in range(1, g))
                    )

    m.optimize()

    status = m.Status
    if m.SolCount == 0:
        return dict(obj=None, optimal=False, time_out=(status in (9,11)),
                    n_vars=m.NumVars, n_constrs=m.NumConstrs, solve_time=m.Runtime)
    return dict(obj=int(round(m.ObjVal)), optimal=(status == 2),
                time_out=(status in (9,11)),
                n_vars=m.NumVars, n_constrs=m.NumConstrs, solve_time=m.Runtime)


# ================================================================ #
#  BaseAlgorithm subclass — CRP-U                                    #
# ================================================================ #

class DeMeloSilva2018BRP(BaseAlgorithm):
    """
    de Melo da Silva et al. (2018) Unrestricted BRP — time-indexed MIP.

    Variant m1 (default): T = UB + N.  Stronger LP relaxation.
    Variant m2: T = UB (relocations only).  More compact, often faster.

    Requires: Gurobi with Named-User Academic License.
    """

    name                = "de Melo da Silva (2018) BRP [Gurobi]"
    category            = "Exact"
    requires_solver     = True
    solver_backend      = "gurobi"
    description         = (
        "Time-indexed MIP for Unrestricted BRP (BRP m1 / m2). "
        "de Melo da Silva, Toulouse & Wolfler Calvo — EJOR 271 (2018). "
        "variant=m1: T=UB+N, stronger LP; variant=m2: T=UB, more compact. "
        "[Requires Gurobi license]"
    )
    compatible_problems = ["CRP-U"]
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
            C    = len(env.containers)
            S    = env.config.num_bays * env.config.num_rows
            H    = env.config.max_tiers
            G    = C  # unique priorities

            T_ub = _minmax_ub(yard, C, S, H)
            if T_ub <= 0:
                T_ub = 1

            t0 = time.perf_counter()

            try:
                import gurobipy  # noqa: F401
            except ImportError:
                print("[DeMeloSilva2018BRP] ERROR: gurobipy not installed.",
                      file=sys.stderr, flush=True)
                self._push(result_queue, step=seed+1, metric=float("inf"),
                           metrics={"relocations": float("inf"), "error": 1.0},
                           progress=(seed+1)/n_seeds)
                continue

            if variant == "m2":
                result = _solve_brp_m2(yard, C, S, H, G, T_ub,
                                       time_limit_s, output_flag)
            else:
                result = _solve_brp_m1(yard, C, S, H, G, T_ub,
                                       time_limit_s, output_flag)

            elapsed  = time.perf_counter() - t0
            n_reloc  = result["obj"] if result["obj"] is not None else -1
            primary  = float(max(n_reloc, 0)) if result["obj"] is not None else float("inf")

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
                f"[DeMeloSilva2018BRP/{variant}] seed={seed}  "
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
                    "m1: T=UB+N; stronger LP relaxation. "
                    "m2: T=UB only; more compact, often faster in practice."
                ),
            },
            "time_limit_s": {
                "type": "float", "default": 3600.0, "min": 1.0, "max": 86400.0,
                "label": "Time limit (s)",
            },
        })
        return base
