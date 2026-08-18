"""
Caserta, Schwarze, Voß (2012)
"A mathematical formulation and complexity considerations for the
blocks relocation problem"
European Journal of Operational Research 219 (2012) 96-104.

Model: BRP-II (Section 3, item-indexed model under Assumption A1)
-------------------------------------------------------------------
Each time period t = 1,...,N corresponds to the *complete* set of moves
(zero or more relocations, followed by exactly one retrieval) needed to
bring block t out of the bay.  Because retrieval order is fixed and
Assumption A1 holds (only blocks stacked above the current target may
be relocated), the horizon T = N is known in advance -- unlike the
move-indexed BRP-I model (see algorithms/CRP_U/exact/solver/
caserta_2012_brp1), which needs an a-priori upper bound on the total
number of moves.

Variables (1-indexed; W = stacks, H = tiers, N = containers)
--------------------------------------------------------------
  b[i,j,n,t]     = 1  if block n occupies slot (i,j) at the start of
                       period t                       (n >= t)
  x[i,j,k,l,n,t] = 1  if block n is relocated from (i,j) to (k,l)
                       during period t                (n > t, j >= 2)
  y[i,j,t]       = 1  if the period's target, block t, is retrieved
                       from (i,j) during period t

``v_nt`` (whether block n has left the bay by the start of period t) is
*not* a decision variable here -- with a fixed retrieval order it is a
plain parameter: v_nt = 1 iff n < t.

Constraints implemented
------------------------
(1)-(3), (6), (7''), (8'), (9), (10) of the paper -- i.e. the complete
BRP-II model, with one necessary correction: the plain literal Eq. (8)
in the 2012 paper only bounds the LIFO-violation sum by 1, which is too
weak whenever more than one block above (i,j) could be disturbed
within the same period; Zehendner, Caserta, Feillet, Schwarze & Voß
(EJOR 245, 2015) show this and fix it with a big-M coefficient
(constraint (8')).  We use their corrected (8') so that the solver
cannot return LIFO-violating (i.e. physically invalid) relocation
sequences -- everything else is implemented exactly as in the 2012
paper, with variable domains left "as is" (e.g. no ``pi_n``-based
pruning of the x-domain, no per-period relocation bound ``UBt``, no
variable pre-fixing).

This module therefore reproduces the *plain* BRP-II model of the 2012
paper (its size matches the paper's own variable-count formula
``(WHN)^2 + 2WHN^2``), as opposed to the improved model BRP-II-A of
Zehendner et al. (2015) -- see
algorithms/CRP_R/exact/solver/zehendner_2015 -- which additionally adds
the per-period bound and a variable pre-processing step to shrink the
model. Comparing the two side by side on the same instance illustrates
the computational benefit of the 2015 refinements.

Note on flow-balance completeness: the state-transition constraint (6a)
must track *every* block still in the bay when moving from period t to
t+1, including the block that becomes the *next* target (n = t+1) --
not just blocks due later (n > t+1). Leaving n = t+1 out (as in a
literal item-by-item transcription) lets the solver place the next
target block "for free" instead of paying for the relocation needed to
uncover it, which was verified with a concrete instance where the
model would otherwise report 0 relocations although >= 2 are provably
necessary. This module includes the full n >= t+1 range.

Requires: Gurobi with a valid license (Named-User Academic or
commercial).
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
#  MinMax heuristic (Section 4 of this same paper) -- used only as   #
#  a cheap "is this instance already trivial?" shortcut               #
# ================================================================ #

def _minmax_ub(stacks: List[List[int]], N: int, W: int, H: int) -> int:
    """Caserta et al. (2012) stack-score heuristic. Feasible upper bound
    on the number of relocations (see algorithms/CRP_R/heuristic/caserta)."""
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
#  Gurobi BRP-II solver                                               #
# ================================================================ #

def _solve_brp_ii(
    initial_pos: Dict[int, Tuple[int, int]],
    N: int,
    W: int,
    H: int,
    time_limit_s: float,
    output_flag: int,
) -> Dict:
    import gurobipy as gp
    from gurobipy import GRB

    T = N  # periods 1..N; model uses t = 1..N-1
    M_big = H  # big-M for the corrected LIFO constraint (8')

    m = gp.Model()
    m.Params.TimeLimit  = float(time_limit_s)
    m.Params.MIPGapAbs  = 0.99
    m.Params.OutputFlag = output_flag

    ws  = range(1, W + 1)
    hs  = range(1, H + 1)

    # ── Helper: initial occupancy ─────────────────────────────── #
    def b_init(i: int, j: int, n: int) -> int:
        return 1 if initial_pos.get(n) == (i, j) else 0

    # ── x active domain: no pi-based pruning -- plain 2012 model ── #
    # x[i,j,k,l,n,t] is active when:
    #   n > t                 (only blocks not yet retrieved can be a blocker)
    #   i != k                 (no reloc to same stack)
    # NOTE: we deliberately do NOT prune x's tier range by N_t(t) (see the
    # comment on b's range below for why an analogous prune on b caused
    # a genuine correctness bug); x has no risky "stale initial value"
    # fallback, so pruning it would only affect performance, not
    # correctness -- but for simplicity and to keep the two variable
    # groups' domains consistent, we use the full 1..H range everywhere.
    def x_active(i, j, k, l, n, t) -> bool:
        if i == k:
            return False
        if n <= t:
            return False
        return True

    # ── Variables ─────────────────────────────────────────────── #
    # b[i,j,n,t]: t=1 is fixed (initial state), so only create for t=2..N
    # (t=N is the trivial single-block state right before retrieving the
    # very last block; needed so constraint (6a) below can properly close
    # the loop for the last "next target" transition).
    #
    # IMPORTANT: tier j always ranges over the FULL 1..H here -- do NOT
    # prune it to "j <= N_t(t)" (containers remaining in the whole bay).
    # That bound is valid on the *sum* of occupied slots, but NOT on any
    # single slot's tier index (an untouched stack can keep blocks at a
    # high tier while other stacks are being cleared out, so N_t(t) is
    # not a safe per-slot bound). Every constraint below that reads a
    # missing b[...] falls back to ``b_init(i, j, n)`` (correct ONLY at
    # t=1); pruning b's domain at t>1 made that fallback silently (and
    # incorrectly) re-assert a block's *original* t=1 position at a
    # later period whenever the pruned tier was skipped -- letting the
    # solver "keep" a block in place for free instead of paying for the
    # relocation actually needed. Verified with a concrete counter-example
    # (2 stacks, 5 containers) where the pruned version was infeasible
    # while the true optimum (4 relocations) requires exactly this slot.
    b: Dict = {}
    for t in range(2, N + 1):
        for i in ws:
            for j in hs:
                for n in range(t, N + 1):
                    b[i, j, n, t] = m.addVar(vtype=GRB.BINARY, name=f"b{i}{j}{n}{t}")

    x: Dict = {}
    for t in range(1, N):
        for n in range(t + 1, N + 1):
            for i in ws:
                for j in range(2, H + 1):
                    for k in ws:
                        if k == i:
                            continue
                        for l in hs:
                            if not x_active(i, j, k, l, n, t):
                                continue
                            x[i, j, k, l, n, t] = m.addVar(
                                vtype=GRB.BINARY, name=f"x{i}{j}{k}{l}{n}{t}"
                            )

    y: Dict = {}
    for t in range(1, N):
        for i in ws:
            for j in hs:
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
    for t in range(1, N + 1):
        for i in ws:
            for j in hs:
                if t == 1:
                    terms_val = sum(b_init(i, j, n) for n in range(1, N + 1))
                    m.addConstr(terms_val <= 1)
                else:
                    m.addConstr(
                        gp.quicksum(b.get((i, j, n, t), b_init(i, j, n))
                                    for n in range(t, N + 1)) <= 1
                    )

    # ── Constraint (3): no gaps in stacks ─────────────────────── #
    for t in range(1, N + 1):
        for i in ws:
            for j in range(1, H):
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

    # ── Constraint (6a): state transition for n > t ───────────── #
    # IMPORTANT: this must cover *every* block not yet due this period,
    # i.e. n = t+1, t+2, ..., N -- NOT just n > t+1.  Exempting n = t+1
    # (the block that becomes the *next* target) leaves its position at
    # period t+1 completely unconstrained by period t's moves, which lets
    # the solver "teleport" it into an already-accessible slot for free
    # instead of paying for the relocation that would legitimately free
    # it.  (Verified against a concrete counter-example: without this,
    # the model can report 0 relocations for an instance that provably
    # needs >= 2.)
    for t in range(1, N):
        for i in ws:
            for j in hs:
                for n in range(t + 1, N + 1):  # every block still in the bay
                    b_curr = b.get((i, j, n, t), b_init(i, j, n))
                    b_next = b.get((i, j, n, t + 1), b_init(i, j, n))
                    m.addConstr(
                        b_next == b_curr + in_flow(i, j, n, t) - out_flow(i, j, n, t)
                    )

    # ── Constraint (6b): block t is at its retrieval position ──── #
    for t in range(1, N):
        for i in ws:
            for j in hs:
                b_val = b.get((i, j, t, t), b_init(i, j, t))
                m.addConstr(b_val == y.get((i, j, t), 0))

    # ── Constraint (7''): exactly one retrieval per period ─────── #
    for t in range(1, N):
        m.addConstr(
            gp.quicksum(y.get((i, j, t), 0) for i in ws for j in hs) == 1
        )

    # ── Constraint (8'): LIFO enforcement (corrected, big-M) ────── #
    # M*(1 - sum_{n>t} x[i,j,k,l,n,t]) >= sum_{n>t} sum_{j'>j} sum_{l'>l} x[i,j',k,l',n,t]
    for t in range(1, N):
        for i in ws:
            for j in range(2, H):      # j < H
                for k in ws:
                    if k == i:
                        continue
                    for l in range(1, H):   # l < H
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

    # ── Constraint (9): Assumption A1 (restricted BRP) ─────────── #
    # sum_{j'<j} y[i,j',t] >= sum_{k,l,n>t} x[i,j,k,l,n,t]
    for t in range(1, N):
        for i in ws:
            for j in range(2, H + 1):
                m.addConstr(
                    gp.quicksum(y.get((i, jp, t), 0) for jp in range(1, j)) >=
                    gp.quicksum(
                        x.get((i, j, k, l, n, t), 0)
                        for k in ws if k != i
                        for l in range(1, H + 1)
                        for n in range(t + 1, N + 1)
                    )
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

class Caserta2012BRPII(BaseAlgorithm):
    """
    Caserta, Schwarze & Voß (2012) -- BRP-II (plain model, Section 3).

    Item-indexed 0-1 IP for the Restricted BRP (Assumption A1).
    T = N periods, each corresponding to the full set of moves needed to
    retrieve one block. Compared to algorithms/CRP_R/exact/solver/
    zehendner_2015 (model BRP-II-A), this module omits the 2015 paper's
    per-period relocation bound and variable pre-processing, staying
    close to the model as originally proposed -- useful as a baseline
    to gauge the impact of those later refinements.

    Solves small-to-medium instances (paper reports up to ~4x6/4x7
    within a reasonable time); larger instances may exceed memory/time
    limits -- see zehendner_2015 for a much more scalable variant.

    Requires: Gurobi with Named-User Academic License.
    """

    name                = "Caserta (2012) BRP-II [Gurobi]"
    category            = "Exact"
    requires_solver     = True
    solver_backend      = "gurobi"
    description         = (
        "BRP-II: item-indexed 0-1 IP for the Restricted BRP (T=N periods). "
        "Caserta, Schwarze & Voß -- EJOR 219 (2012). Plain model (LIFO "
        "constraint corrected per Zehendner et al. 2015 for validity; no "
        "per-period bound / pre-processing). [Requires Gurobi license]"
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
                print("[Caserta2012BRPII] ERROR: gurobipy not installed.",
                      file=sys.stderr, flush=True)
                self._push(result_queue, step=seed+1, metric=float("inf"),
                           metrics={"relocations": float("inf"), "error": 1.0},
                           progress=(seed+1)/n_seeds)
                continue

            stacks      = _stacks_as_lists(yard)
            initial_pos = _initial_layout(yard, W)

            # Cheap shortcut: layout already sorted → 0 relocations needed.
            heuristic_ub = _minmax_ub(stacks, N, W, H)
            if heuristic_ub == 0:
                metrics = {
                    "relocations": 0.0, "steps": 0.0, "time": 0.0,
                    "optimal_proven": 1.0, "time_out": 0.0, "n_vars": 0.0,
                    "n_constrs": 0.0, "solve_time_s": 0.0, "feasible": 1.0,
                    "heuristic_ub": 0.0,
                }
                all_metrics.append(metrics)
                self._push(result_queue, step=seed+1, metric=0.0,
                           metrics=metrics, progress=(seed+1)/n_seeds,
                           snapshot=env.get_state_snapshot())
                continue

            result = _solve_brp_ii(initial_pos, N, W, H, time_limit_s, output_flag)

            elapsed = time.perf_counter() - t0
            feasible = result["obj"] is not None
            n_reloc  = result["obj"] if feasible else None
            primary  = float(n_reloc) if feasible else float("inf")

            if primary < self._best_metric:
                self._best_metric   = primary
                self._best_solution = []

            metrics = {
                "relocations":    float(n_reloc) if feasible else float("inf"),
                "steps":          float(n_reloc) if feasible else float("inf"),
                "time":           float(n_reloc) if feasible else float("inf"),
                "optimal_proven": 1.0 if result["optimal"] else 0.0,
                "time_out":       1.0 if result["time_out"] else 0.0,
                "n_vars":         float(result["n_vars"]),
                "n_constrs":      float(result["n_constrs"]),
                "solve_time_s":   round(elapsed, 4),
                "feasible":       1.0 if feasible else 0.0,
                "heuristic_ub":   float(heuristic_ub),
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
                f"[Caserta2012BRPII] seed={seed}  N={N} W={W} H={H}  "
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
            "time_limit_s": {
                "type": "float", "default": 3600.0, "min": 1.0, "max": 86400.0,
                "label": "Time limit (s)",
                "help": "Per-instance Gurobi time limit. Paper uses 60 min (3600 s) or 1 day (86400 s).",
            },
        })
        return base
