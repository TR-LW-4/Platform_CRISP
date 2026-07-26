"""
Tang, Jiang, Liu, Dong (2015)
"Research into container reshuffling and stacking problems in container
terminal yards"
IIE Transactions 47 (2015) 751-766.

Model: ILP (the "improved model" of Section 3.2)
-------------------------------------------------
Improves the Wan, Liu & Tsai (2009) MRIP model (see
algorithms/CRP_R/exact/solver/wan_2009) for the same static BRP (CRP-R)
by removing the column-relationship variables u[s,i], v[s,i], z[s,i] and
the five constraint sets used only to derive them. Whether container i is
in the same column as -- and above -- the container s being retrieved is
instead determined directly, one column at a time, by two per-column
constraints (paper's (2)-(3)) that bound the reshuffle indicator y[s,i].

Assumptions (same as MRIP, Section 3 of the paper):
  A1. Anticipatory restriction: a container is reshuffled iff it is at a
      higher tier of the SAME COLUMN as the container being retrieved.
      -> Equivalent to platform's CRP-R (Assumption A1 / restricted BRP).
  A2. Multiple-moves restriction: each reshuffled container is moved once
      only per retrieval stage.

Paper notation -> platform mapping
  S containers  -> N (container count)
  C columns     -> W = num_bays x num_rows (stacks)
  P positions   -> H = max_tiers

Variables (1-indexed)
  x[s,i,c,p]  = 1  if container i is at (col c, tier p) at stage s
  y[s,i]      = 1  if container i is reshuffled in the retrieval of s
  w[s,i,j]    = 1  if both i,j are reshuffled at stage s and j is above i

(No u[s,i], v[s,i], z[s,i] -- this is precisely the reduction the paper
introduces relative to MRIP.)

Objective: min Sum_{s=1}^{N-1} Sum_{i=s+1}^{N} y[s,i]   (paper's Eq. (1),
identical to MRIP's objective -- only the constraints that pin down y
change.)

y-defining constraints (paper's (2)-(3), one pair per column c):
  (1 - Sum_p x[s,s,c,p])*P + y[s,i] >= (Sum_p p*x[s,i,c,p]
                                        - Sum_p p*x[s,s,c,p]) / P
  (Sum_p p*x[s,s,c,p] - Sum_p p*x[s,i,c,p]) / P <= 1 - y[s,i]
These force y[s,i] = 1 exactly when i sits in the same column as s and at
a higher tier at the start of stage s -- i.e. i genuinely blocks s -- and
force y[s,i] = 0 otherwise. All remaining constraints (slot occupancy, no
floating containers, column-change-on-reshuffle, pairwise ordering of
co-reshuffled containers via w, position preservation for non-reshuffled
containers, and initial/pre-processing fixing of x) are unchanged from
MRIP, just re-derived without reference to u, v, z.

Effect of the reduction (paper, end of Section 3.2): removing u, v, z
saves 3*N^2 binary variables at the cost of (C-4)*N*(N-1) extra
constraints; the paper reports this model reaches optimality in roughly
1/4 to 1/3 of the MRIP solve time on the hardest tested class (6 columns,
5 tiers, 100% utilisation).

ILP-DK heuristic (k > 0)
  Rolling-horizon variant used for the dynamic experiments in the paper
  (Table 5, "ILP model-DK" for K = 5, 6, 7, 8): at each retrieval step,
  solve a K-stage sub-ILP for the next K containers and execute the
  first step's decisions. k = 0 -> exact full ILP.

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
    Implements the paper's pre-processing fixing of x[s,i,c,p] for
    s <= si[i] (its Eq. (16)).
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
#  Core ILP solver (full or K-stage submodel)                        #
# ================================================================ #

def _solve_ilp_core(
    initial_pos: Dict[int, Tuple[int, int]],
    N: int,
    C: int,
    H: int,
    n_stages: int,
    time_limit_s: float,
    output_flag: int,
) -> Optional[Dict]:
    """
    Build and solve the ILP (or its K-stage submodel).

    Parameters
    ----------
    initial_pos : {priority: (col, tier)} 1-indexed for N containers
    N           : number of containers (priorities 1..N)
    C           : number of columns
    H           : max tiers per column  (paper's P)
    n_stages    : number of retrieval stages to model (<= N-1)
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

    # -- x variables (only for stages beyond pre-processing depth) -- #
    x_var: Dict = {}
    for s in range(2, n_stages + 2):  # need state after last retrieval
        if s > N:
            break
        for i in range(max(s, 1), N + 1):
            if s > si.get(i, i):
                for c in range(1, C + 1):
                    for p in range(1, P + 1):
                        x_var[s, i, c, p] = m.addVar(vtype=GRB.BINARY)

    # -- y, w variables (no u, v, z -- this is the paper's reduction) -- #
    y: Dict = {}; w: Dict = {}
    for s in range(1, n_stages + 1):
        for i in range(s + 1, N + 1):
            y[s, i] = m.addVar(vtype=GRB.BINARY)
            for j in range(s + 1, N + 1):
                if j != i:
                    w[s, i, j] = m.addVar(vtype=GRB.BINARY)

    m.update()

    # -- Helper aggregates -------------------------------------------- #
    def slot_sum(s: int, i: int):
        return gp.quicksum(get_x(s, i, c, p)
                           for c in range(1, C+1) for p in range(1, P+1))

    def col_p_sum(s: int, i: int, c: int):
        """Sum_p x[s,i,c,p] -- presence of container i in column c at stage s."""
        return gp.quicksum(get_x(s, i, c, p) for p in range(1, P+1))

    def col_pp_sum(s: int, i: int, c: int):
        """Sum_p p*x[s,i,c,p] -- position of container i in column c."""
        return gp.quicksum(p * get_x(s, i, c, p) for p in range(1, P+1))

    def pos_sum(s: int, i: int):
        """Sum over ALL columns -- position of i regardless of column."""
        return gp.quicksum(p * get_x(s, i, c, p)
                           for c in range(1, C+1) for p in range(1, P+1))

    # -- Objective (identical to MRIP, paper's Eq. (1)) --------------- #
    m.setObjective(
        gp.quicksum(y[s, i]
                    for s in range(1, n_stages + 1)
                    for i in range(s + 1, N + 1)),
        gp.GRB.MINIMIZE,
    )

    # -- Constraints ---------------------------------------------------- #
    for s in range(1, n_stages + 1):
        for i in range(s + 1, N + 1):
            # (2)-(3): per-column definition of y[s,i], replacing MRIP's
            # u/v/z-based derivation. y[s,i] is forced to 1 iff there is
            # some column c containing both s and i with i above s, and
            # forced to 0 otherwise.
            for c in range(1, C + 1):
                cps_s = col_p_sum(s, s, c)
                cpp_s = col_pp_sum(s, s, c)
                cpp_i = col_pp_sum(s, i, c)
                m.addConstr(
                    (1 - cps_s) * P + y[s, i] >= (cpp_i - cpp_s) / P
                )
                m.addConstr(
                    (cpp_s - cpp_i) / P <= 1 - y[s, i]
                )

        # (4): each container at exactly one slot
        for i in range(s, N + 1):
            m.addConstr(slot_sum(s, i) == 1)

        # (5): at most one container per slot
        for c in range(1, C + 1):
            for p in range(1, P + 1):
                m.addConstr(
                    gp.quicksum(get_x(s, i, c, p) for i in range(s, N + 1)) <= 1
                )

        # (6): no floating (containers must rest on ground or lower container)
        for c in range(1, C + 1):
            for p in range(2, P + 1):
                m.addConstr(
                    gp.quicksum(get_x(s, i, c, p) for i in range(s, N + 1)) <=
                    gp.quicksum(get_x(s, i, c, p-1) for i in range(s, N + 1))
                )

        # Constraints that link stage s to s+1
        if s < n_stages + 1 and s + 1 <= N:
            for i in range(s + 1, N + 1):
                # (7): reshuffled container cannot stay in same column as s
                for c in range(1, C + 1):
                    m.addConstr(
                        col_p_sum(s + 1, i, c) <=
                        2 - y[s, i] - col_p_sum(s, s, c)
                    )
                # (13)-(14): non-reshuffled containers keep positions
                for c in range(1, C + 1):
                    for p in range(1, P + 1):
                        xi_s   = get_x(s, i, c, p)
                        xi_s1  = get_x(s + 1, i, c, p)
                        m.addConstr(xi_s1 - xi_s >= -y[s, i])
                        m.addConstr(xi_s  - xi_s1 >= -y[s, i])

            # (8)-(12): pairwise ordering of reshuffled containers
            for i in range(s + 1, N + 1):
                ps_i = pos_sum(s, i)
                for j in range(s + 1, N + 1):
                    if j == i:
                        continue
                    ps_j = pos_sum(s, j)
                    # (8): P*(2-y_i-y_j+w_ij) >= ps_j - ps_i
                    m.addConstr(
                        P * (2 - y[s, i] - y[s, j] + w[s, i, j]) >= ps_j - ps_i
                    )
                    # (9): P*(y_i+y_j+w_ij-3) <= ps_j - ps_i
                    m.addConstr(
                        P * (y[s, i] + y[s, j] + w[s, i, j] - 3) <= ps_j - ps_i
                    )
                    # (10): w <= y_i
                    m.addConstr(w[s, i, j] <= y[s, i])
                    # (11): w <= y_j
                    m.addConstr(w[s, i, j] <= y[s, j])
                    # (12): per-column relative height flip after reshuffling
                    if s + 1 <= N:
                        for c in range(1, C + 1):
                            m.addConstr(
                                col_pp_sum(s+1, i, c) - col_pp_sum(s+1, j, c) >=
                                -P*(1 - w[s, i, j])
                                -P*(1 - y[s, i])
                                -P*(1 - y[s, j])
                                -P*(1 - col_p_sum(s+1, i, c))
                            )

    # -- Solve ------------------------------------------------------------ #
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
#  Full ILP (exact) and ILP-DK (rolling horizon)                     #
# ================================================================ #

def _run_full_ilp(
    yard: Yard,
    N: int,
    C: int,
    H: int,
    time_limit_s: float,
    output_flag: int,
) -> Dict:
    initial_pos = _extract_layout(yard)
    result = _solve_ilp_core(
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
    ILP-DK rolling horizon heuristic (paper Table 5, "ILP model-DK").
    At each retrieval step, solve a K-stage sub-ILP and execute
    the first step's reshuffling decisions.
    """
    current_pos = _extract_layout(yard)
    total_reloc = 0
    total_time  = 0.0
    all_vars    = 0
    all_constrs = 0
    time_per_step = time_limit_s / max(N - 1, 1)

    remaining = sorted(current_pos.keys())

    for step in range(N - 1):
        n_rem = len(remaining)
        if n_rem <= 1:
            break

        global_to_local = {g: l + 1 for l, g in enumerate(remaining)}
        local_to_global = {l + 1: g for l, g in enumerate(remaining)}

        local_pos = {global_to_local[g]: current_pos[g] for g in remaining}

        K_eff = min(k, n_rem - 1) if k > 0 else n_rem - 1

        t0 = time.perf_counter()
        result = _solve_ilp_core(
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

            for local_i, (c, p) in result["x_next"].items():
                global_i = local_to_global.get(local_i)
                if global_i is not None:
                    current_pos[global_i] = (c, p)

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

class Tang2015ILP(BaseAlgorithm):
    """
    Tang, Jiang, Liu & Dong (2015) -- ILP / ILP-DK for Restricted BRP.

    Improves Wan et al. (2009)'s MRIP model (algorithms/CRP_R/exact/
    solver/wan_2009) for the same static BRP (CRP-R) by removing the
    column-relationship variables u, v, z and deriving the reshuffle
    indicator y[s,i] directly from two per-column constraints instead.

    Two modes selected via ``extra["k"]``:
      k = 0  -> exact full ILP  (optimal; same objective as MRIP, solves
                faster -- paper reports ~1/4-1/3 of MRIP's time on the
                hardest tested class)
      k >= 1 -> ILP-DK rolling-horizon heuristic (fast near-optimal,
                mirrors the paper's dynamic-simulation experiments)

    Paper benchmarks (Table 1, 100% utilisation, CPLEX 11.0):
      (6 cols, 4 tiers, 21 containers): ILP avg 50.88 s   (MRIP: 135.36 s)
      (6 cols, 5 tiers, 26 containers): ILP solves 58% of instances to
        optimality within 1 h vs. MRIP's 48%.

    Historical note: Galle (2018) further reduces the variable count via
    a binary encoding of positions; that model is already in this
    platform (algorithms/CRP_R/exact/solver/galle_2018).

    Requires: Gurobi with Named-User Academic License.
    """

    name                = "Tang et al. (2015) ILP [Gurobi]"
    category            = "Exact"
    requires_solver     = True
    solver_backend      = "gurobi"
    description         = (
        "ILP / ILP-DK: improved IP model for Restricted BRP, derived from "
        "Wan (2009) MRIP by removing column-relationship variables u,v,z. "
        "Tang, Jiang, Liu & Dong -- IIE Trans. 47 (2015). "
        "k=0: exact (faster than MRIP); k>=1: rolling-horizon heuristic "
        "with K-step lookahead. [Requires Gurobi license]"
    )
    compatible_problems = ["CRP-R"]
    step_label          = "Seed"

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
        n_seeds      = max(1, cfg.num_eval_seeds)
        all_metrics: List[Dict] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.config.seed = seed
            env.reset(seed=seed, options={"skip_auto_retrieve": True})

            yard  = env.yard
            N     = len(env.containers)
            C     = env.config.num_bays * env.config.num_rows
            H     = env.config.max_tiers

            t0 = time.perf_counter()

            try:
                import gurobipy  # noqa: F401
            except ImportError:
                print("[Tang2015ILP] ERROR: gurobipy not installed.",
                      file=sys.stderr, flush=True)
                self._push(result_queue, step=seed+1, metric=float("inf"),
                           metrics={"relocations": float("inf"), "error": 1.0},
                           progress=(seed+1)/n_seeds)
                continue

            use_exact = (k <= 0 or k >= N - 1)
            if use_exact:
                result = _run_full_ilp(yard, N, C, H, time_limit_s, output_flag)
            else:
                result = _run_rolling_horizon(yard, N, C, H, k, time_limit_s, output_flag)

            elapsed  = time.perf_counter() - t0
            n_reloc  = result["obj"] if result["obj"] is not None else -1
            primary  = float(max(n_reloc, 0)) if result["obj"] is not None else float("inf")

            if primary < self._best_metric:
                self._best_metric   = primary
                self._best_solution = []

            mode_str = "ILP(exact)" if use_exact else f"ILP{k}(rolling)"
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
                f"[Tang2015ILP/{mode_str}] seed={seed}  N={N} C={C} H={H}  "
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
            "num_eval_seeds": {
                "type": "int", "default": 1, "min": 1, "max": 20,
                "label": "Seeds",
            },
            "k": {
                "type": "int", "default": 0, "min": 0, "max": 100,
                "label": "k  (0 = exact ILP; >=1 = ILP-DK rolling horizon)",
                "help": (
                    "k=0: solve full ILP to optimality. "
                    "k=1..N-1: rolling horizon -- at each retrieval step solve a "
                    "k-stage sub-ILP; much faster, near-optimal. "
                    "Paper's dynamic experiments use k=5..8 (Table 5)."
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
