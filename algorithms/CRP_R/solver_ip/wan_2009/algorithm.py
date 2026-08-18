"""
Wan, Liu, Tsai (2009)
"The Assignment of Storage Locations to Containers for a Container Stack"
Naval Research Logistics 56 (2009) 699–713.

Model: MRIP (Minimum Reshuffles Integer Program)
-------------------------------------------------
First IP formulation for the static BRP (CRP-R).  Tracks the evolution of
stack configurations across retrieval stages as a function of reshuffling
decisions.

Assumptions (Section 3, paper):
  A1. Anticipatory restriction: a container is reshuffled iff it is at a
      higher tier of the SAME COLUMN as the container being retrieved.
      → Equivalent to platform's CRP-R (Assumption A1 / restricted BRP).
  A2. Multiple-moves restriction: each reshuffled container is moved once
      only per retrieval stage.

Paper notation → platform mapping
  S containers  → N (container count)
  C columns     → W = num_bays × num_rows (stacks)
  P positions   → H = max_tiers

Variables (1-indexed)
  x[s,i,c,p]  = 1  if container i is at (col c, tier p) at stage s
  u[s,i]      = 1  if col(i) ≥ col(s) at stage s          (column indicator)
  v[s,i]      = 1  if col(i) ≤ col(s) at stage s
  z[s,i]      = 1  if containers i and s are in the same column at stage s
  y[s,i]      = 1  if container i is reshuffled in the retrieval of s
  w[s,i,j]    = 1  if both i,j are reshuffled at stage s and j is above i

Objective: min Σ_{s=1}^{N-1} Σ_{i=s+1}^{N} y[s,i]

Constraints (7)–(29) from the paper, implemented with big-P instead of P.

MRIPK heuristic (k > 0)
  Rolling-horizon variant: at each retrieval step, solve a K-stage subMRIP
  for the next K containers and execute the first step's decisions.
  k = 0 → exact full MRIP.

Requires: Gurobi with valid license (Named-User Academic or commercial).
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
#  Layout helpers                                                     #
# ================================================================ #

def _extract_layout(yard: Yard) -> Dict[int, Tuple[int, int]]:
    """Return {priority: (col 1-indexed, tier 1-indexed)}."""
    pos: Dict[int, Tuple[int, int]] = {}
    stack_list = sorted(yard.stacks.values(), key=lambda s: (s.bay, s.row))
    for col, stack in enumerate(stack_list, 1):
        for tier, cont in enumerate(stack.containers, 1):
            pos[cont.priority] = (col, tier)
    return pos


def _compute_si(initial_pos: Dict[int, Tuple[int, int]], N: int) -> Dict[int, int]:
    """
    si[i] = smallest priority among containers in the same column
            and at or below the initial tier of container i.
    This is the first retrieval stage where container i gets moved
    (either reshuffled or retrieved directly).
    Implements Constraint (27) pre-processing from the paper.
    """
    si: Dict[int, int] = {}
    for i in range(1, N + 1):
        if i not in initial_pos:
            si[i] = i
            continue
        c_i, p_i = initial_pos[i]
        min_below = i
        for j in range(1, i):
            if j in initial_pos:
                c_j, p_j = initial_pos[j]
                if c_j == c_i and p_j < p_i:
                    min_below = min(min_below, j)
        si[i] = min_below
    return si


# ================================================================ #
#  Core MRIP solver (full or K-stage submodel)                       #
# ================================================================ #

def _solve_mrip_core(
    initial_pos: Dict[int, Tuple[int, int]],
    N: int,
    C: int,
    H: int,
    n_stages: int,
    time_limit_s: float,
    output_flag: int,
) -> Optional[Dict]:
    """
    Build and solve the MRIP or MRIPK submodel.

    Parameters
    ----------
    initial_pos : {priority: (col, tier)} 1-indexed for N containers
    N           : number of containers (priorities 1..N)
    C           : number of columns
    H           : max tiers per column  (paper's P)
    n_stages    : number of retrieval stages to model (≤ N-1)
    time_limit_s: Gurobi time limit
    output_flag : 0=silent, 1=verbose

    Returns
    -------
    dict with keys: obj, optimal, time_out, n_vars, n_constrs, solve_time,
                    y_vals {(s,i): 0/1}, x_next {i: (col,tier)} after stage 1
    or None if infeasible / error.
    """
    import gurobipy as gp
    from gurobipy import GRB

    P = H  # paper's P = platform's H

    si = _compute_si(initial_pos, N)

    def X1(i: int, c: int, p: int) -> int:
        return 1 if initial_pos.get(i) == (c, p) else 0

    def get_x(s: int, i: int, c: int, p: int):
        """Return constant (int) or Gurobi variable for x[s,i,c,p]."""
        if s <= si.get(i, i):
            return X1(i, c, p)
        key = (s, i, c, p)
        return x_var.get(key, X1(i, c, p))

    m = gp.Model()
    m.Params.TimeLimit  = float(time_limit_s)
    m.Params.MIPGapAbs  = 0.99
    m.Params.OutputFlag = output_flag

    # ── x variables (only for stages beyond pre-processing depth) ── #
    x_var: Dict = {}
    for s in range(2, n_stages + 2):  # need state after last retrieval
        if s > N:
            break
        for i in range(max(s, 1), N + 1):
            if s > si.get(i, i):
                for c in range(1, C + 1):
                    for p in range(1, P + 1):
                        x_var[s, i, c, p] = m.addVar(vtype=GRB.BINARY)

    # ── u, v, z, y, w variables ──────────────────────────────────── #
    u: Dict = {}; v: Dict = {}; z: Dict = {}
    y: Dict = {}; w: Dict = {}
    for s in range(1, n_stages + 1):
        for i in range(s + 1, N + 1):
            u[s, i] = m.addVar(vtype=GRB.BINARY)
            v[s, i] = m.addVar(vtype=GRB.BINARY)
            z[s, i] = m.addVar(vtype=GRB.BINARY)
            y[s, i] = m.addVar(vtype=GRB.BINARY)
            for j in range(s + 1, N + 1):
                if j != i:
                    w[s, i, j] = m.addVar(vtype=GRB.BINARY)

    m.update()

    # ── Helper aggregates ─────────────────────────────────────────── #
    def col_sum(s: int, i: int):
        return gp.quicksum(c * get_x(s, i, c, p)
                           for c in range(1, C+1) for p in range(1, P+1))

    def pos_sum(s: int, i: int):
        return gp.quicksum(p * get_x(s, i, c, p)
                           for c in range(1, C+1) for p in range(1, P+1))

    def slot_sum(s: int, i: int):
        return gp.quicksum(get_x(s, i, c, p)
                           for c in range(1, C+1) for p in range(1, P+1))

    def col_p_sum(s: int, i: int, c: int):
        """Σ_p x[s,i,c,p] — presence of container i in column c at stage s."""
        return gp.quicksum(get_x(s, i, c, p) for p in range(1, P+1))

    def col_pp_sum(s: int, i: int, c: int):
        """Σ_p p*x[s,i,c,p] — position of container i in column c."""
        return gp.quicksum(p * get_x(s, i, c, p) for p in range(1, P+1))

    # ── Objective ─────────────────────────────────────────────────── #
    m.setObjective(
        gp.quicksum(y[s, i]
                    for s in range(1, n_stages + 1)
                    for i in range(s + 1, N + 1)),
        gp.GRB.MINIMIZE,
    )

    # ── Constraints ──────────────────────────────────────────────── #
    for s in range(1, n_stages + 1):
        cs_s = col_sum(s, s)
        ps_s = pos_sum(s, s)

        for i in range(s + 1, N + 1):
            cs_i = col_sum(s, i)
            ps_i = pos_sum(s, i)

            # (7)-(8): define u[s,i] (col_i ≥ col_s ?)
            m.addConstr(C * u[s, i] >= cs_i - cs_s + 1)
            m.addConstr(C * u[s, i] - C <= cs_i - cs_s)
            # (9)-(10): define v[s,i]
            m.addConstr(C * v[s, i] >= cs_s - cs_i + 1)
            m.addConstr(C * v[s, i] - C <= cs_s - cs_i)
            # (11): z[s,i] = u+v-1
            m.addConstr(z[s, i] == u[s, i] + v[s, i] - 1)
            # (12): P*y ≥ P*z - P + ps_i - ps_s
            m.addConstr(P * y[s, i] >= P * z[s, i] - P + ps_i - ps_s)
            # (13): y ≤ z
            m.addConstr(y[s, i] <= z[s, i])
            # (14): ps_s - ps_i ≤ P*(1-y)
            m.addConstr(ps_s - ps_i <= P * (1 - y[s, i]))

        # (15): each container at exactly one slot
        for i in range(s, N + 1):
            m.addConstr(slot_sum(s, i) == 1)

        # (16): at most one container per slot
        for c in range(1, C + 1):
            for p in range(1, P + 1):
                m.addConstr(
                    gp.quicksum(get_x(s, i, c, p) for i in range(s, N + 1)) <= 1
                )

        # (17): no floating (containers must rest on ground or lower container)
        for c in range(1, C + 1):
            for p in range(2, P + 1):
                m.addConstr(
                    gp.quicksum(get_x(s, i, c, p) for i in range(s, N + 1)) <=
                    gp.quicksum(get_x(s, i, c, p-1) for i in range(s, N + 1))
                )

        # Constraints that link stage s to s+1
        if s < n_stages + 1 and s + 1 <= N:
            for i in range(s + 1, N + 1):
                # (18): reshuffled container cannot stay in same column as s
                for c in range(1, C + 1):
                    m.addConstr(
                        col_p_sum(s + 1, i, c) <=
                        2 - y[s, i] - col_p_sum(s, s, c)
                    )
                # (24)-(25): non-reshuffled containers keep positions
                for c in range(1, C + 1):
                    for p in range(1, P + 1):
                        xi_s   = get_x(s, i, c, p)
                        xi_s1  = get_x(s + 1, i, c, p)
                        m.addConstr(xi_s1 - xi_s >= -y[s, i])
                        m.addConstr(xi_s  - xi_s1 >= -y[s, i])

            # (19)-(23): pairwise ordering of reshuffled containers
            for i in range(s + 1, N + 1):
                ps_i = pos_sum(s, i)
                for j in range(s + 1, N + 1):
                    if j == i:
                        continue
                    ps_j = pos_sum(s, j)
                    # (19): P*(2-y_i-y_j+w_ij) ≥ ps_j - ps_i
                    m.addConstr(
                        P * (2 - y[s, i] - y[s, j] + w[s, i, j]) >= ps_j - ps_i
                    )
                    # (20): P*(y_i+y_j+w_ij-3) ≤ ps_j - ps_i
                    m.addConstr(
                        P * (y[s, i] + y[s, j] + w[s, i, j] - 3) <= ps_j - ps_i
                    )
                    # (21): w ≤ y_i
                    m.addConstr(w[s, i, j] <= y[s, i])
                    # (22): w ≤ y_j
                    m.addConstr(w[s, i, j] <= y[s, j])
                    # (23): per-column relative height flip after reshuffling
                    if s + 1 <= N:
                        for c in range(1, C + 1):
                            m.addConstr(
                                col_pp_sum(s+1, i, c) - col_pp_sum(s+1, j, c) >=
                                -P*(1 - w[s, i, j])
                                -P*(1 - y[s, i])
                                -P*(1 - y[s, j])
                                -P*(1 - col_p_sum(s+1, i, c))
                            )

    # ── Solve ─────────────────────────────────────────────────────── #
    m.optimize()

    status = m.Status
    if m.SolCount == 0:
        return None

    obj_val = int(round(m.ObjVal))

    # Extract y decisions
    y_vals = {(s, i): round(var.X) for (s, i), var in y.items()}

    # Extract container positions after stage 1 (for rolling horizon)
    x_next: Dict[int, Tuple[int, int]] = {}
    s_next = 2
    for i in range(s_next, N + 1):
        for c in range(1, C + 1):
            for p in range(1, P + 1):
                val = get_x(s_next, i, c, p)
                v_val = val.X if hasattr(val, "X") else val
                if v_val > 0.5:
                    x_next[i] = (c, p)

    return dict(
        obj=obj_val,
        optimal=(status == GRB.OPTIMAL),
        time_out=(status in (9, 11)),
        n_vars=m.NumVars,
        n_constrs=m.NumConstrs,
        solve_time=m.Runtime,
        y_vals=y_vals,
        x_next=x_next,
    )


# ================================================================ #
#  Full MRIP (exact) and MRIPK (rolling horizon)                     #
# ================================================================ #

def _run_full_mrip(
    yard: Yard,
    N: int,
    C: int,
    H: int,
    time_limit_s: float,
    output_flag: int,
) -> Dict:
    initial_pos = _extract_layout(yard)
    result = _solve_mrip_core(
        initial_pos, N, C, H,
        n_stages=N - 1,
        time_limit_s=time_limit_s,
        output_flag=output_flag,
    )
    if result is None:
        return dict(obj=None, optimal=False, time_out=False,
                    n_vars=0, n_constrs=0, solve_time=0.0)
    return result


def _run_rolling_horizon(
    yard: Yard,
    N: int,
    C: int,
    H: int,
    k: int,
    time_limit_s: float,
    output_flag: int,
) -> Dict:
    """
    MRIPK rolling horizon heuristic.
    At each retrieval step, solve a K-stage subMRIP and execute
    the first step's reshuffling decisions.
    """
    # Build current config as {local_priority: (col, tier)}
    current_pos = _extract_layout(yard)
    total_reloc = 0
    total_time  = 0.0
    all_vars    = 0
    all_constrs = 0
    time_per_step = time_limit_s / max(N - 1, 1)

    # Remaining containers by priority (global priorities still unused)
    remaining = sorted(current_pos.keys())

    for step in range(N - 1):
        n_rem = len(remaining)
        if n_rem <= 1:
            break

        # Re-index remaining containers 1..n_rem
        global_to_local = {g: l + 1 for l, g in enumerate(remaining)}
        local_to_global = {l + 1: g for l, g in enumerate(remaining)}

        local_pos = {global_to_local[g]: current_pos[g] for g in remaining}

        K_eff = min(k, n_rem - 1) if k > 0 else n_rem - 1

        t0 = time.perf_counter()
        result = _solve_mrip_core(
            local_pos, n_rem, C, H,
            n_stages=K_eff,
            time_limit_s=time_per_step,
            output_flag=output_flag,
        )
        total_time += time.perf_counter() - t0

        if result is None:
            # Fallback: count blockers above target as relocations
            target_g = remaining[0]
            tc, tp = current_pos[target_g]
            for g in remaining[1:]:
                gc, gp_ = current_pos[g]
                if gc == tc and gp_ > tp:
                    total_reloc += 1
        else:
            total_reloc += result["obj"] if K_eff == n_rem - 1 else sum(
                result["y_vals"].get((1, li), 0) for li in range(2, n_rem + 1)
            )
            all_vars    += result["n_vars"]
            all_constrs += result["n_constrs"]

            # Update positions for containers 2..n_rem (local) from x_next
            for local_i, (c, p) in result["x_next"].items():
                global_i = local_to_global.get(local_i)
                if global_i is not None:
                    current_pos[global_i] = (c, p)

        # Retrieve the smallest-priority container
        retrieved = remaining.pop(0)
        del current_pos[retrieved]

    return dict(
        obj=total_reloc,
        optimal=False,
        time_out=(total_time >= time_limit_s),
        n_vars=all_vars,
        n_constrs=all_constrs,
        solve_time=total_time,
    )


# ================================================================ #
#  BaseAlgorithm subclass                                            #
# ================================================================ #

class Wan2009MRIP(BaseAlgorithm):
    """
    Wan, Liu & Tsai (2009) — MRIP / MRIPK for Restricted BRP.

    First IP formulation for the static BRP (CRP-R).

    Two modes selected via ``extra["k"]``:
      k = 0  → exact full MRIP  (optimal; slow for large N)
      k ≥ 1  → MRIPK rolling-horizon heuristic (fast near-optimal)

    Paper benchmarks (Table 2, heavy density):
      (6 cols, 4 tiers, 17 containers): MRIP avg 85.97 s
      (6 cols, 5 tiers, 21 containers): MRIP avg 689.42 s

    Historical note: Tang et al. (2015) improved MRIP by removing the
    column-relationship variables u, v, z; Galle (2018) reduced variables
    further via binary encoding. Both are already in this platform.

    Requires: Gurobi with Named-User Academic License.
    """

    name                = "Wan (2009) MRIP [Gurobi]"
    category            = "Exact"
    requires_solver     = True
    solver_backend      = "gurobi"
    description         = (
        "MRIP / MRIPK: first IP model for Restricted BRP. "
        "Wan, Liu & Tsai — NRL 56 (2009). "
        "k=0: exact; k≥1: rolling-horizon heuristic with K-step lookahead. "
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

        k            = int(extra.get("k", 0))
        time_limit_s = float(extra.get("time_limit_s", 3600.0))
        output_flag  = int(extra.get("output_flag", 0))
        n_seeds = 1  # multi-seed eval removed; single run only
        all_metrics: List[Dict] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.reset(seed=seed, options={"skip_auto_retrieve": True})

            yard  = env.yard
            N     = len(env.containers)
            C     = env.config.num_bays * env.config.num_rows
            H     = env.config.max_tiers

            t0 = time.perf_counter()

            try:
                import gurobipy  # noqa: F401
            except ImportError:
                print("[Wan2009MRIP] ERROR: gurobipy not installed.",
                      file=sys.stderr, flush=True)
                self._push(result_queue, step=seed+1, metric=float("inf"),
                           metrics={"relocations": float("inf"), "error": 1.0},
                           progress=(seed+1)/n_seeds)
                continue

            # Choose mode
            use_exact = (k <= 0 or k >= N - 1)
            if use_exact:
                result = _run_full_mrip(yard, N, C, H, time_limit_s, output_flag)
            else:
                result = _run_rolling_horizon(yard, N, C, H, k, time_limit_s, output_flag)

            elapsed  = time.perf_counter() - t0
            n_reloc  = result["obj"] if result["obj"] is not None else -1
            primary  = float(max(n_reloc, 0)) if result["obj"] is not None else float("inf")

            if primary < self._best_metric:
                self._best_metric   = primary
                self._best_solution = []

            mode_str = "MRIP(exact)" if use_exact else f"MRIP{k}(rolling)"
            metrics: Dict = {
                "relocations":    float(max(n_reloc, 0)),
                "steps":          float(max(n_reloc, 0)),
                "time":           float(max(n_reloc, 0)),
                "optimal_proven": 1.0 if result.get("optimal", False) else 0.0,
                "time_out":       1.0 if result.get("time_out", False) else 0.0,
                "n_vars":         float(result.get("n_vars", 0)),
                "n_constrs":      float(result.get("n_constrs", 0)),
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
                f"[Wan2009MRIP/{mode_str}] seed={seed}  N={N} C={C} H={H}  "
                f"reloc={n_reloc}  opt={result.get('optimal', False)}  "
                f"vars={result.get('n_vars', 0)}  t={elapsed:.2f}s",
                file=sys.stderr, flush=True,
            )

        if all_metrics:
            agg = {k2: float(np.mean([mm[k2] for mm in all_metrics if k2 in mm]))
                   for k2 in all_metrics[0]}
            self._push(result_queue, step=n_seeds, metric=self._best_metric,
                       metrics=agg, progress=1.0)

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "k": {
                "type": "int", "default": 0, "min": 0, "max": 100,
                "label": "k  (0 = exact MRIP; ≥1 = MRIPK rolling horizon)",
                "help": (
                    "k=0: solve full MRIP to optimality. "
                    "k=1..N-1: rolling horizon — at each retrieval step solve a "
                    "k-stage subMRIP; much faster, near-optimal. "
                    "Paper recommends k=4..6 for 6-column stacks."
                ),
            },
            "time_limit_s": {
                "type": "float", "default": 3600.0, "min": 1.0, "max": 86400.0,
                "label": "Time limit (s)",
                "help": (
                    "For k=0 (exact): total solve time limit. "
                    "For k>0 (rolling): per-step budget = time_limit / (N-1)."
                ),
            },
        })
        return base
