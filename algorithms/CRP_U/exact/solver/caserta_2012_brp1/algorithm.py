"""
Caserta, Schwarze, Voß (2012)
"A mathematical formulation and complexity considerations for the
blocks relocation problem"
European Journal of Operational Research 219 (2012) 96-104.

Model: BRP-I (Section 3, move-indexed model -- no Assumption A1)
--------------------------------------------------------------------
Unlike BRP-II (algorithms/CRP_R/exact/solver/caserta_2012_brp2, which
assumes only blocks above the *current target* may be relocated), BRP-I
maps the *complete* feasible region of the BRP: at every time period at
most one single move happens (a relocation of the top container of any
stack, or a retrieval), and that move may originate from *any* stack --
matching CRP-U ("unrestricted relocation") in this platform.

Because a period now corresponds to a single move rather than to "all
moves needed to retrieve one block", the total number of periods T
(= total number of moves) is *not* known a priori and must be bounded
from above.  We use the tight, instance-specific bound produced by the
MinMax heuristic of Section 4 of this same paper (used elsewhere in the
platform as `algorithms/CRP_R/heuristic/caserta`) plus N retrievals, or
-- optionally -- the paper's own worst-case guarantee from Lemma 1
(Section 2): for 2 <= W <= N,
    k  = ceil((N-1)/(W-1))
    UB = k(N-1) - k(k-1)/2 * (W-1)
(UB = N-1 for W > N).  Either bound is individually valid (a proven
upper bound on the optimal relocation count), so T = min(both) + 1 (the
"+1" leaves one idle period of slack) is always large enough to contain
an optimal solution.

Variables (1-indexed; W = stacks, H = tiers, N = containers, T = periods)
---------------------------------------------------------------------------
  b[i,j,n,t] = 1  if block n is in slot (i,j) at the start of period t
  v[n,t]     = 1  if block n has already been retrieved by some period
                   t0 in {1,...,t-1}          (DECISION variable here --
                   contrast with BRP-II, where it is a fixed parameter)
  x[i,j,k,l,n,t] = 1  if block n is relocated from (i,j) to (k,l)
                       during period t
  y[i,j,n,t] = 1  if block n is retrieved from (i,j) during period t

Constraints implemented: (1)-(7) of the paper, verbatim (no Assumption
A1, no additional pruning of the x/y domains beyond the structurally
necessary tier-height bounds).  Objective: maximise sum_t v[N,t]
(equivalently: minimise the period at which the last block leaves the
bay, i.e. the total number of moves used).

WARNING -- variable-count explosion.  The paper reports this model
generating ``2*W*H*N*T + (W*H)^2*N*T + N*T`` binary variables; even a
tiny 3x5,N=9 instance with T=20 needs > 46,000 variables, and the paper
itself could not solve instances beyond ~3x3 in the allotted time. This
module is therefore only practical for very small toy instances. A hard
guard (``extra["max_vars"]``, default 200,000) aborts before building
an unreasonably large model.

Requires: Gurobi with a valid license (Named-User Academic or
commercial).
"""

from __future__ import annotations

import math
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
    """Return {priority: (stack_idx, tier_idx)} -- both 1-indexed."""
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
#  Upper bounds on the number of relocations                        #
# ================================================================ #

def _minmax_ub(stacks: List[List[int]], N: int, W: int, H: int) -> int:
    """Caserta et al. (2012) stack-score heuristic (Section 4). Feasible
    upper bound on the number of relocations for THIS instance."""
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


def _lemma1_ub(N: int, W: int) -> int:
    """
    Worst-case upper bound on the number of relocations, Lemma 1
    (Section 2 of the paper).  Valid for ANY instance with the given
    N, W (and H >= N-1); does not look at the actual layout.
    """
    if N <= 1:
        return 0
    if W > N:
        return N - 1
    W_eff = max(W, 2)
    k = math.ceil((N - 1) / (W_eff - 1))
    ub = k * (N - 1) - (k * (k - 1) // 2) * (W_eff - 1)
    return max(ub, 0)


# ================================================================ #
#  Gurobi BRP-I solver                                                #
# ================================================================ #

def _solve_brp_i(
    initial_pos: Dict[int, Tuple[int, int]],
    N: int,
    W: int,
    H: int,
    T: int,
    time_limit_s: float,
    output_flag: int,
) -> Dict:
    import gurobipy as gp
    from gurobipy import GRB

    m = gp.Model()
    m.Params.TimeLimit  = float(time_limit_s)
    m.Params.MIPGapAbs  = 0.99
    m.Params.OutputFlag = output_flag

    ws = range(1, W + 1)
    hs = range(1, H + 1)
    ns = range(1, N + 1)
    ts = range(1, T + 1)

    def b_init(i: int, j: int, n: int) -> int:
        return 1 if initial_pos.get(n) == (i, j) else 0

    # ── Variables ─────────────────────────────────────────────── #
    # b[i,j,n,t] for t=2..T  (t=1 fixed by the initial layout via b_init)
    b: Dict = {}
    for t in range(2, T + 1):
        for i in ws:
            for j in hs:
                for n in ns:
                    b[i, j, n, t] = m.addVar(vtype=GRB.BINARY, name=f"b{i}_{j}_{n}_{t}")

    # v[n,t] for t=2..T  (v[n,1] == 0 by definition, not created)
    v: Dict = {}
    for t in range(2, T + 1):
        for n in ns:
            v[n, t] = m.addVar(vtype=GRB.BINARY, name=f"v{n}_{t}")

    # x[i,j,k,l,n,t] / y[i,j,n,t] for t=1..T-1 (a move at t updates b at t+1)
    x: Dict = {}
    for t in range(1, T):
        for i in ws:
            for j in hs:
                for k in ws:
                    if k == i:
                        continue
                    for l in hs:
                        for n in ns:
                            x[i, j, k, l, n, t] = m.addVar(
                                vtype=GRB.BINARY, name=f"x{i}_{j}_{k}_{l}_{n}_{t}"
                            )

    y: Dict = {}
    for t in range(1, T):
        for i in ws:
            for j in hs:
                for n in ns:
                    y[i, j, n, t] = m.addVar(
                        vtype=GRB.BINARY, name=f"y{i}_{j}_{n}_{t}"
                    )

    m.update()

    def b_get(i, j, n, t):
        if t == 1:
            return b_init(i, j, n)
        return b.get((i, j, n, t), 0)

    def v_get(n, t):
        if t == 1:
            return 0
        return v.get((n, t), 0)

    def x_get(i, j, k, l, n, t):
        return x.get((i, j, k, l, n, t), 0)

    def y_get(i, j, n, t):
        return y.get((i, j, n, t), 0)

    # ── Objective (paper): maximise sum_t v[N,t] ───────────────── #
    m.setObjective(
        gp.quicksum(v_get(N, t) for t in range(2, T + 1)),
        GRB.MAXIMIZE,
    )

    # ── (1) each block is either in the bay or already retrieved ── #
    for n in ns:
        for t in range(2, T + 1):
            m.addConstr(
                gp.quicksum(b_get(i, j, n, t) for i in ws for j in hs) + v_get(n, t) == 1
            )

    # ── (2) at most one block per slot ─────────────────────────── #
    for i in ws:
        for j in hs:
            for t in range(2, T + 1):
                m.addConstr(gp.quicksum(b_get(i, j, n, t) for n in ns) <= 1)

    # ── (3) no gaps within a stack ──────────────────────────────── #
    for i in ws:
        for j in range(1, H):
            for t in range(2, T + 1):
                m.addConstr(
                    gp.quicksum(b_get(i, j, n, t) for n in ns) >=
                    gp.quicksum(b_get(i, j + 1, n, t) for n in ns)
                )

    # ── (4) at most one move per period ─────────────────────────── #
    for t in range(1, T):
        m.addConstr(
            gp.quicksum(x_get(i, j, k, l, n, t)
                        for i in ws for j in hs for k in ws if k != i
                        for l in hs for n in ns) +
            gp.quicksum(y_get(i, j, n, t) for i in ws for j in hs for n in ns)
            <= 1
        )

    # ── (5) retrieval order ─────────────────────────────────────── #
    for n in range(1, N):
        m.addConstr(
            gp.quicksum(v_get(n, t) for t in ts) >=
            gp.quicksum(v_get(n + 1, t) for t in ts) + 1
        )

    # ── (6) flow balance ────────────────────────────────────────── #
    for i in ws:
        for j in hs:
            for n in ns:
                for t in range(2, T + 1):
                    m.addConstr(
                        b_get(i, j, n, t) ==
                        b_get(i, j, n, t - 1)
                        + gp.quicksum(x_get(k, l, i, j, n, t - 1) for k in ws if k != i for l in hs)
                        - gp.quicksum(x_get(i, j, k, l, n, t - 1) for k in ws if k != i for l in hs)
                        - y_get(i, j, n, t - 1)
                    )

    # ── (7) relation between v (outside) and moving (y) variables ── #
    for n in ns:
        for t in range(2, T + 1):
            m.addConstr(
                v_get(n, t) ==
                gp.quicksum(y_get(i, j, n, t0)
                            for i in ws for j in hs for t0 in range(1, t))
            )

    # ── Solve ────────────────────────────────────────────────────── #
    m.optimize()

    status = m.Status
    if m.SolCount == 0:
        return dict(obj=None, optimal=False, time_out=(status in (9, 11)),
                    n_vars=m.NumVars, n_constrs=m.NumConstrs, solve_time=m.Runtime)

    obj_val      = m.ObjVal
    total_moves  = int(round(T - obj_val))
    relocations  = max(total_moves - N, 0)
    return dict(obj=relocations, total_moves=total_moves,
                optimal=(status == 2), time_out=(status in (9, 11)),
                n_vars=m.NumVars, n_constrs=m.NumConstrs, solve_time=m.Runtime)


def _estimate_var_count(N: int, W: int, H: int, T: int) -> int:
    """2*W*H*N*T + (W*H)^2*N*T + N*T, per the paper's own formula."""
    return 2 * W * H * N * T + (W * H) ** 2 * N * T + N * T


# ================================================================ #
#  BaseAlgorithm subclass                                            #
# ================================================================ #

class Caserta2012BRPI(BaseAlgorithm):
    """
    Caserta, Schwarze & Voß (2012) -- BRP-I (complete model, Section 3).

    Move-indexed 0-1 IP mapping the FULL feasible region of the BRP:
    no Assumption A1, i.e. the top container of *any* stack may be
    relocated at each step -- matching CRP-U ("unrestricted relocation")
    in this platform.  Contrast with algorithms/CRP_R/exact/solver/
    caserta_2012_brp2 (model BRP-II), which restricts moves to blockers
    of the current target and is far more tractable as a result.

    Only usable on very small toy instances (the paper itself reports
    failures beyond ~3x3); a hard variable-count guard aborts before
    building unreasonably large models.

    Requires: Gurobi with Named-User Academic License.
    """

    name                = "Caserta (2012) BRP-I [Gurobi]"
    category            = "Exact"
    requires_solver     = True
    solver_backend      = "gurobi"
    description         = (
        "BRP-I: move-indexed 0-1 IP mapping the COMPLETE feasible region "
        "of the BRP (no Assumption A1). Caserta, Schwarze & Voß -- EJOR "
        "219 (2012). Needs an a-priori bound T on total moves; only "
        "practical for tiny instances. [Requires Gurobi license]"
    )
    compatible_problems = ["CRP-U"]
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

        time_limit_s = float(extra.get("time_limit_s", 3600.0))
        output_flag  = int(extra.get("output_flag", 0))
        ub_method    = str(extra.get("ub_method", "heuristic"))
        max_vars     = int(extra.get("max_vars", 200_000))
        n_seeds      = max(1, cfg.num_eval_seeds)
        all_metrics: List[Dict] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.config.seed = seed
            env.reset(seed=seed, options={"skip_auto_retrieve": True})

            yard = env.yard
            N    = len(env.containers)
            W    = env.config.num_bays * env.config.num_rows
            H    = env.config.max_tiers

            t0 = time.perf_counter()

            try:
                import gurobipy  # noqa: F401
            except ImportError:
                print("[Caserta2012BRPI] ERROR: gurobipy not installed.",
                      file=sys.stderr, flush=True)
                self._push(result_queue, step=seed+1, metric=float("inf"),
                           metrics={"relocations": float("inf"), "error": 1.0},
                           progress=(seed+1)/n_seeds)
                continue

            stacks      = _stacks_as_lists(yard)
            initial_pos = _initial_layout(yard, W)

            reloc_ub_heur = _minmax_ub(stacks, N, W, H)
            if reloc_ub_heur == 0:
                metrics = {
                    "relocations": 0.0, "steps": 0.0, "time": 0.0,
                    "optimal_proven": 1.0, "time_out": 0.0, "n_vars": 0.0,
                    "n_constrs": 0.0, "solve_time_s": 0.0, "feasible": 1.0,
                    "T": 0.0, "aborted_too_large": 0.0,
                }
                all_metrics.append(metrics)
                self._push(result_queue, step=seed+1, metric=0.0,
                           metrics=metrics, progress=(seed+1)/n_seeds,
                           snapshot=env.get_state_snapshot())
                continue

            reloc_ub_lemma1 = _lemma1_ub(N, W)
            if ub_method == "lemma1":
                reloc_ub = reloc_ub_lemma1
            else:
                # min of the two: both are individually valid upper bounds
                # on the optimal relocation count, so their min still is.
                reloc_ub = min(reloc_ub_heur, reloc_ub_lemma1) if reloc_ub_lemma1 > 0 else reloc_ub_heur

            T = reloc_ub + N + 1  # "+1": one idle period of slack

            n_vars_est = _estimate_var_count(N, W, H, T)
            if n_vars_est > max_vars:
                print(
                    f"[Caserta2012BRPI] seed={seed}  ABORTED -- estimated "
                    f"{n_vars_est:,} binary variables exceeds max_vars="
                    f"{max_vars:,}. Try a smaller instance or raise "
                    f"extra['max_vars'].",
                    file=sys.stderr, flush=True,
                )
                metrics = {
                    "relocations": float("inf"), "steps": float("inf"),
                    "time": float("inf"), "optimal_proven": 0.0,
                    "time_out": 0.0, "n_vars": float(n_vars_est),
                    "n_constrs": 0.0, "solve_time_s": 0.0, "feasible": 0.0,
                    "T": float(T), "aborted_too_large": 1.0,
                }
                all_metrics.append(metrics)
                self._push(result_queue, step=seed+1, metric=float("inf"),
                           metrics=metrics, progress=(seed+1)/n_seeds)
                continue

            result = _solve_brp_i(initial_pos, N, W, H, T, time_limit_s, output_flag)

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
                "T":              float(T),
                "aborted_too_large": 0.0,
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
                f"[Caserta2012BRPI] seed={seed}  N={N} W={W} H={H} T={T}  "
                f"reloc={n_reloc}  opt={result['optimal']}  "
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
            "num_eval_seeds": {
                "type": "int", "default": 1, "min": 1, "max": 20,
                "label": "Seeds",
            },
            "time_limit_s": {
                "type": "float", "default": 3600.0, "min": 1.0, "max": 86400.0,
                "label": "Time limit (s)",
            },
            "ub_method": {
                "type": "str", "default": "heuristic", "options": ["heuristic", "lemma1"],
                "label": "Horizon T method",
                "help": (
                    "heuristic: T = MinMax-heuristic relocation count + N + 1 "
                    "(instance-specific, usually tight). "
                    "lemma1: T = worst-case Lemma-1 bound + N + 1 "
                    "(instance-independent, always larger, matches the "
                    "paper's Section 2 guarantee)."
                ),
            },
            "max_vars": {
                "type": "int", "default": 200_000, "min": 1_000, "max": 5_000_000,
                "label": "Max binary variables",
                "help": "Abort before building the model if the estimated "
                        "variable count exceeds this (BRP-I is only "
                        "practical for tiny instances).",
            },
        })
        return base
