"""
Zehendner, Caserta, Feillet, Schwarze, Voß (2015)
"An improved mathematical formulation for the blocks relocation problem"
European Journal of Operational Research 245 (2015) 415–422.

Model: BRP-II-A
----------------
Correction and improvement of the BRP-II model from Caserta et al. (2012).
Adds:
  - Corrected Constraint (8') with big-M
  - Constraint (A'): Assumption A1 (only blocks above target may be relocated)
  - Constraint (B): per-period upper bound UBt on relocations
  - Pre-processing step: fix ~65–89 % of variables using initial layout info

Variables (1-indexed; W = stacks, H = tiers, N = containers, T = N periods)
---------------------------------------------------------------------------
  b[i,j,n,t]     = 1  if block n is at position (i,j) at start of period t
                       defined for n ≥ t, t = 1..T-1
  x[i,j,k,l,n,t] = 1  if block n is relocated from (i,j) to (k,l) in period t
                       defined for n > t, j ≥ 2, i ≠ k, t = 1..T-1
  y[i,j,t]       = 1  if block t is retrieved from (i,j) in period t
                       t = 1..T-1

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

def _initial_layout(yard: Yard, W: int) -> Dict[int, Tuple[int, int]]:
    """Return {priority: (stack_idx, tier_idx)} — both 1-indexed."""
    pos: Dict[int, Tuple[int, int]] = {}
    stack_list = sorted(yard.stacks.values(), key=lambda s: (s.bay, s.row))
    for s_idx, stack in enumerate(stack_list, 1):
        for h_idx, cont in enumerate(stack.containers, 1):  # h=1=bottom
            pos[cont.priority] = (s_idx, h_idx)
    return pos


def _stacks_as_lists(yard: Yard) -> List[List[int]]:
    """Return list of priority lists (bottom-to-top), 0-indexed stacks."""
    stack_list = sorted(yard.stacks.values(), key=lambda s: (s.bay, s.row))
    return [[c.priority for c in st.containers] for st in stack_list]


# ================================================================ #
#  π_n: first period when block n is moved                           #
# ================================================================ #

def _compute_pi(
    initial_pos: Dict[int, Tuple[int, int]],
    stacks: List[List[int]],
    N: int,
) -> Dict[int, int]:
    """
    π_n = min priority among blocks in the same stack at or below block n.
    Eq. (21) in the paper.
    If π_n < n: block n is first relocated in period π_n.
    If π_n = n: block n is retrieved directly (never relocated).
    """
    pi: Dict[int, int] = {}
    for n in range(1, N + 1):
        if n not in initial_pos:
            pi[n] = n
            continue
        i_n, j_n = initial_pos[n]          # 1-indexed
        stack = stacks[i_n - 1]            # 0-indexed list
        # blocks in same stack at tiers 1..j_n (0..j_n-1 in list)
        below_incl = stack[:j_n]
        pi[n] = min(below_incl) if below_incl else n
    return pi


# ================================================================ #
#  MinMax heuristic — upper bound on relocations                     #
# ================================================================ #

def _minmax_ub(stacks: List[List[int]], N: int, W: int, H: int) -> int:
    """Caserta (2012) MinMax heuristic. Returns upper bound on relocations."""
    bay = [list(s) for s in stacks]
    height = [len(s) for s in bay]

    def min_p(si: int) -> int:
        return min(bay[si]) if bay[si] else N + 1

    mp_ = [min_p(s) for s in range(W)]
    n_reloc = 0

    for tp in range(1, N + 1):
        ts = ti = None
        for si in range(W):
            for i, p in enumerate(bay[si]):
                if p == tp:
                    ts, ti = si, i
                    break
            if ts is not None:
                break
        if ts is None:
            continue
        while len(bay[ts]) > ti + 1:
            n_reloc += 1
            r = bay[ts][-1]
            good = [s for s in range(W) if s != ts and height[s] < H and mp_[s] > r]
            dst = (min(good, key=lambda s: mp_[s]) if good
                   else max((s for s in range(W) if s != ts and height[s] < H),
                            key=lambda s: mp_[s], default=None))
            if dst is None:
                break
            bay[ts].pop(); height[ts] -= 1
            bay[dst].append(r); height[dst] += 1
            mp_[dst] = min_p(dst); mp_[ts] = min_p(ts)
        bay[ts] = [p for p in bay[ts] if p != tp]
        height[ts] -= 1
        mp_[ts] = min_p(ts)

    return n_reloc


# ================================================================ #
#  Zhu et al. (2012) lower bound — LBt and LBt+                     #
# ================================================================ #

def _zhu_lb(stacks: List[List[int]], N: int, W: int, H: int):
    """
    Compute Zhu (2012) LB components.
    Returns (LBt, LBtplus) as lists, index 1..N.
    """
    bay = [list(s) for s in stacks]
    counted: set = set()
    LBt    = [0] * (N + 1)
    LBtplus = [0] * (N + 1)

    for t in range(1, N + 1):
        ts = th = None
        for si, stack in enumerate(bay):
            for hi, p in enumerate(stack):
                if p == t:
                    ts, th = si, hi
                    break
            if ts is not None:
                break
        if ts is None:
            continue

        above = bay[ts][th + 1:]
        for p in above:
            if p not in counted:
                LBt[t] += 1
                counted.add(p)

        for p in above:
            can_place = False
            for si2, stack2 in enumerate(bay):
                if si2 == ts or len(stack2) >= H:
                    continue
                if not stack2 or min(stack2) > p:
                    can_place = True
                    break
            if not can_place:
                LBtplus[t] += 1

        # Remove t and blocks above it from bay (t retrieved, above relocated)
        bay[ts] = [x for x in bay[ts] if x != t and x not in above]

    return LBt, LBtplus


# ================================================================ #
#  Per-period upper bound UBt                                        #
# ================================================================ #

def _per_period_ub(
    LBt: List[int],
    LBtplus: List[int],
    UB: int,
    N: int,
    H: int,
) -> Dict[int, int]:
    """
    UBt = min(Rt, H-1) where Rt per Eq. (14).
    Falls back to H-1 when Rt would be negative (heuristic is already optimal).
    """
    UBt: Dict[int, int] = {}
    for t in range(1, N):
        Rt = (UB - 1
              - sum(LBt[k] for k in range(1, t))
              - sum(LBt[k] for k in range(t + 1, N + 1))
              - sum(LBtplus[k] for k in range(t, N + 1)))
        UBt[t] = min(max(Rt, 0), H - 1)
    return UBt


# ================================================================ #
#  Gurobi BRP-II-A solver                                            #
# ================================================================ #

def _solve_brp_ii_a(
    initial_pos: Dict[int, Tuple[int, int]],
    pi: Dict[int, int],
    N: int,
    W: int,
    H: int,
    UBt: Dict[int, int],
    time_limit_s: float,
    output_flag: int,
) -> Dict:
    import gurobipy as gp
    from gurobipy import GRB

    T = N  # periods 1..N; model uses t = 1..N-1
    M_big = H  # big-M for constraint (8')

    m = gp.Model()
    m.Params.TimeLimit  = float(time_limit_s)
    m.Params.MIPGapAbs  = 0.99
    m.Params.OutputFlag = output_flag

    ws  = range(1, W + 1)
    hs  = range(1, H + 1)
    hs2 = range(2, H + 1)  # tiers ≥ 2 (source of relocations)

    # ── Helper: initial occupancy ─────────────────────────────── #
    def b_init(i: int, j: int, n: int) -> int:
        return 1 if initial_pos.get(n) == (i, j) else 0

    # ── N_t: containers remaining at start of period t ────────── #
    def N_t(t: int) -> int:
        return N + 1 - t

    # ── Pre-processing mask for x ──────────────────────────────── #
    # x[i,j,k,l,n,t] is active when:
    #   t >= pi[n]            (not before first move of n)
    #   j <= N_t(t)           (source tier height bound)
    #   l <= N_t(t) - 1       (dest tier height bound: can only go to top)
    #   i != k                (no reloc to same stack)

    def x_active(i, j, k, l, n, t) -> bool:
        if i == k:
            return False
        if t < pi.get(n, n):
            return False
        if j > N_t(t) or l > N_t(t) - 1:
            return False
        return True

    # ── Variables ─────────────────────────────────────────────── #
    # b[i,j,n,t]: period t=1 is fixed (initial state), so we only
    # need b for t=2..T-1 where the value is not determined by PP.
    # For simplicity we create all and fix PP ones via constraints.

    # Create b for t=2..N-1 (t=1 is fixed by initial state constraint)
    b: Dict = {}
    for t in range(2, N):           # t=1 will be handled by constraints
        Nt = N_t(t)
        for i in ws:
            for j in range(1, min(H, Nt) + 1):
                for n in range(t, N + 1):
                    b[i, j, n, t] = m.addVar(vtype=GRB.BINARY, name=f"b{i}{j}{n}{t}")

    # Create x
    x: Dict = {}
    for t in range(1, N):
        Nt = N_t(t)
        for n in range(t + 1, N + 1):
            for i in ws:
                for j in range(2, min(H, Nt) + 1):
                    for k in ws:
                        if k == i:
                            continue
                        for l in range(1, min(H, Nt) + 1):
                            if not x_active(i, j, k, l, n, t):
                                continue
                            x[i, j, k, l, n, t] = m.addVar(
                                vtype=GRB.BINARY, name=f"x{i}{j}{k}{l}{n}{t}"
                            )

    # Create y[i,j,t] for t=1..N-1
    y: Dict = {}
    for t in range(1, N):
        Nt = N_t(t)
        for i in ws:
            for j in range(1, min(H, Nt) + 1):
                y[i, j, t] = m.addVar(vtype=GRB.BINARY, name=f"y{i}{j}{t}")

    m.update()

    # ── Objective: minimise total relocations ─────────────────── #
    m.setObjective(
        gp.quicksum(x.get((i, j, k, l, n, t), 0)
                    for t in range(1, N)
                    for n in range(t + 1, N + 1)
                    for i in ws
                    for j in range(2, H + 1)
                    for k in ws if k != i
                    for l in range(1, H + 1)
                    if (i, j, k, l, n, t) in x),
        GRB.MINIMIZE,
    )

    # ── Helper: sum of x going INTO (i,j) in period t for block n ── #
    def in_flow(i, j, n, t):
        return gp.quicksum(
            x[k2, l2, i, j, n, t]
            for k2 in ws for l2 in range(2, H + 1)
            if (k2, l2, i, j, n, t) in x
        )

    def out_flow(i, j, n, t):
        return gp.quicksum(
            x[i, j, k2, l2, n, t]
            for k2 in ws if k2 != i
            for l2 in range(1, H + 1)
            if (i, j, k2, l2, n, t) in x
        )

    # ── Constraint (2): at most one block per slot ────────────── #
    for t in range(1, N):
        Nt = N_t(t)
        for i in ws:
            for j in range(1, min(H, Nt) + 1):
                terms = []
                if t == 1:
                    # Use initial values
                    terms_val = sum(b_init(i, j, n) for n in range(1, N + 1))
                    m.addConstr(terms_val <= 1)
                else:
                    m.addConstr(
                        gp.quicksum(b.get((i, j, n, t), b_init(i, j, n))
                                    for n in range(t, N + 1)) <= 1
                    )

    # ── Constraint (3): no gaps in stacks ─────────────────────── #
    for t in range(1, N):
        Nt = N_t(t)
        for i in ws:
            for j in range(1, min(H - 1, Nt) + 1):
                if t == 1:
                    lhs = sum(b_init(i, j, n) for n in range(1, N + 1))
                    rhs = sum(b_init(i, j + 1, n) for n in range(1, N + 1))
                    if lhs < rhs:
                        m.addConstr(0 >= 1)  # infeasible if violated at t=1
                else:
                    m.addConstr(
                        gp.quicksum(b.get((i, j, n, t), b_init(i, j, n))
                                    for n in range(t, N + 1)) >=
                        gp.quicksum(b.get((i, j + 1, n, t), b_init(i, j + 1, n))
                                    for n in range(t, N + 1))
                    )

    # ── Pre-processing: fix b for periods where block hasn't moved ── #
    for n in range(1, N + 1):
        pi_n = pi.get(n, n)
        for t in range(2, min(pi_n, N) + 1):  # b fixed for t=2..pi_n
            if (t - 1) >= N:
                continue
            Nt = N_t(t)
            for i in ws:
                for j in range(1, min(H, Nt) + 1):
                    if (i, j, n, t) in b:
                        m.addConstr(b[i, j, n, t] == b_init(i, j, n))

    # ── Constraint (6a): state transition for n > t ───────────── #
    for t in range(1, N - 1):
        Nt = N_t(t)
        for i in ws:
            for j in range(1, min(H, Nt) + 1):
                for n in range(t + 2, N + 1):  # n > t+1 (n is not the new target)
                    b_curr = b.get((i, j, n, t), b_init(i, j, n))
                    b_next = b.get((i, j, n, t + 1), b_init(i, j, n))
                    m.addConstr(
                        b_next == b_curr + in_flow(i, j, n, t) - out_flow(i, j, n, t)
                    )

    # ── Constraint (6b): block t is at its retrieval position ──── #
    for t in range(1, N):
        Nt = N_t(t)
        for i in ws:
            for j in range(1, min(H, Nt) + 1):
                b_val = b.get((i, j, t, t), b_init(i, j, t))
                m.addConstr(b_val == y.get((i, j, t), 0))

    # ── Constraint (7''): exactly one retrieval per period ─────── #
    for t in range(1, N):
        m.addConstr(
            gp.quicksum(y.get((i, j, t), 0) for i in ws for j in hs) == 1
        )

    # ── Constraint (8'): LIFO enforcement ─────────────────────── #
    # M*(1 - sum_{n>t} x[i,j,k,l,n,t]) >= sum_{n>t} sum_{j'>j} sum_{l'>l} x[i,j',k,l',n,t]
    for t in range(1, N):
        Nt = N_t(t)
        for i in ws:
            for j in range(2, min(H, Nt)):      # j < H
                for k in ws:
                    if k == i:
                        continue
                    for l in range(1, min(H, Nt)):   # l < H
                        xijkl = gp.quicksum(
                            x.get((i, j, k, l, n, t), 0)
                            for n in range(t + 1, N + 1)
                        )
                        above = gp.quicksum(
                            x.get((i, jp, k, lp, n, t), 0)
                            for n in range(t + 1, N + 1)
                            for jp in range(j + 1, H + 1)
                            for lp in range(l + 1, H + 1)
                        )
                        m.addConstr(M_big * (1 - xijkl) >= above)

    # ── Constraint (A'): Assumption A1 (restricted BRP) ────────── #
    # sum_{j'<j} y[i,j',t] >= sum_{k,l,n>t} x[i,j,k,l,n,t]
    for t in range(1, N):
        Nt = N_t(t)
        for i in ws:
            for j in range(2, min(H, Nt) + 1):
                m.addConstr(
                    gp.quicksum(y.get((i, jp, t), 0) for jp in range(1, j)) >=
                    gp.quicksum(
                        x.get((i, j, k, l, n, t), 0)
                        for k in ws if k != i
                        for l in range(1, H + 1)
                        for n in range(t + 1, N + 1)
                    )
                )

    # ── Constraint (B): per-period relocation upper bound ─────── #
    for t in range(1, N):
        ub_t = UBt.get(t, H - 1)
        if ub_t <= 0:
            # No relocations allowed in period t → feasible only if target is on top
            m.addConstr(
                gp.quicksum(
                    x.get((i, j, k, l, n, t), 0)
                    for i in ws for j in range(2, H + 1)
                    for k in ws if k != i
                    for l in range(1, H + 1)
                    for n in range(t + 1, N + 1)
                ) == 0
            )
        else:
            m.addConstr(
                gp.quicksum(
                    x.get((i, j, k, l, n, t), 0)
                    for i in ws for j in range(2, H + 1)
                    for k in ws if k != i
                    for l in range(1, H + 1)
                    for n in range(t + 1, N + 1)
                ) <= ub_t
            )

    # ── Solve ─────────────────────────────────────────────────── #
    m.optimize()

    status = m.Status
    if m.SolCount == 0:
        return dict(obj=None, optimal=False, time_out=(status in (9, 11)),
                    n_vars=m.NumVars, n_constrs=m.NumConstrs,
                    solve_time=m.Runtime)
    obj = int(round(m.ObjVal))
    return dict(obj=obj, optimal=(status == 2),
                time_out=(status in (9, 11)),
                n_vars=m.NumVars, n_constrs=m.NumConstrs,
                solve_time=m.Runtime)


# ================================================================ #
#  BaseAlgorithm subclass                                            #
# ================================================================ #

class Zehendner2015BRPIIA(BaseAlgorithm):
    """
    Zehendner, Caserta, Feillet, Schwarze & Voß (2015) — BRP-II-A.

    Corrects and tightens the BRP-II model of Caserta et al. (2012):
      • Per-period relocation bound UBt = min(Rt, H-1)
      • Pre-processing: fixes ~65–89 % of variables
      • Constraint (A'): restricted BRP (Assumption A1)

    Solves instances with up to ~28 containers (5–4 class) efficiently.
    Larger instances may exceed memory limits.

    Requires: Gurobi with Named-User Academic License.
    """

    name                = "Zehendner (2015) BRP-II-A [Gurobi]"
    category            = "Exact"
    requires_solver     = True
    solver_backend      = "gurobi"
    description         = (
        "BRP-II-A: corrected & improved BRP-II IP for Restricted BRP. "
        "Zehendner, Caserta, Feillet, Schwarze & Voß — EJOR 245 (2015). "
        "Per-period UBt + pre-processing. [Requires Gurobi license]"
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
            W     = env.config.num_bays * env.config.num_rows
            H     = env.config.max_tiers

            t0 = time.perf_counter()

            try:
                import gurobipy  # noqa: F401
            except ImportError:
                print("[Zehendner2015BRPIIA] ERROR: gurobipy not installed.",
                      file=sys.stderr, flush=True)
                self._push(result_queue, step=seed+1, metric=float("inf"),
                           metrics={"relocations": float("inf"), "error": 1.0},
                           progress=(seed+1)/n_seeds)
                continue

            # Layout and π_n
            stacks      = _stacks_as_lists(yard)
            initial_pos = _initial_layout(yard, W)
            pi          = _compute_pi(initial_pos, stacks, N)

            # UB via MinMax heuristic
            UB = _minmax_ub(stacks, N, W, H)
            if UB == 0:
                # Already sorted — trivial
                metrics = {
                    "relocations": 0.0, "steps": 0.0, "time": 0.0,
                    "optimal_proven": 1.0, "time_out": 0.0, "n_vars": 0.0,
                    "n_constrs": 0.0, "solve_time_s": 0.0, "feasible": 1.0,
                    "UB": 0.0,
                }
                all_metrics.append(metrics)
                self._push(result_queue, step=seed+1, metric=0.0,
                           metrics=metrics, progress=(seed+1)/n_seeds,
                           snapshot=env.get_state_snapshot())
                continue

            # LB via Zhu (2012) and per-period UBt
            LBt, LBtplus = _zhu_lb(stacks, N, W, H)
            UBt = _per_period_ub(LBt, LBtplus, UB, N, H)

            result = _solve_brp_ii_a(
                initial_pos, pi, N, W, H, UBt,
                time_limit_s, output_flag,
            )

            elapsed = time.perf_counter() - t0
            n_reloc = result["obj"] if result["obj"] is not None else -1
            primary = float(max(n_reloc, 0)) if result["obj"] is not None else float("inf")

            if primary < self._best_metric:
                self._best_metric   = primary
                self._best_solution = []

            metrics = {
                "relocations":    float(max(n_reloc, 0)),
                "steps":          float(max(n_reloc, 0)),
                "time":           float(max(n_reloc, 0)),
                "optimal_proven": 1.0 if result["optimal"] else 0.0,
                "time_out":       1.0 if result["time_out"] else 0.0,
                "n_vars":         float(result["n_vars"]),
                "n_constrs":      float(result["n_constrs"]),
                "solve_time_s":   round(elapsed, 4),
                "feasible":       0.0 if result["obj"] is None else 1.0,
                "UB":             float(UB),
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
                f"[Zehendner2015BRPIIA] seed={seed}  C={N} W={W} H={H}  "
                f"UB={UB}  reloc={n_reloc}  opt={result['optimal']}  "
                f"vars={result['n_vars']}  t={elapsed:.2f}s",
                file=sys.stderr, flush=True,
            )

        if all_metrics:
            agg = {k: float(np.mean([mm[k] for mm in all_metrics if k in mm]))
                   for k in all_metrics[0]}
            self._push(result_queue, step=n_seeds, metric=self._best_metric,
                       metrics=agg, progress=1.0)

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "time_limit_s": {
                "type": "float", "default": 3600.0, "min": 1.0, "max": 86400.0,
                "label": "Time limit (s)",
                "help": "Per-instance Gurobi time limit. Paper uses 60 min (3600 s).",
            },
        })
        return base
