"""
BRP-II Gurobi formulation for Caserta2012BRPII.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from core.yard import Yard
from algorithms.CRP_R.solver_ip.common import gurobi_on


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
    on the number of relocations (see algorithms/CRP_R/heuristic/caserta_2012)."""
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
                    solve_time=m.Runtime, x_vals=(), y_vals=())
    obj = int(round(m.ObjVal))
    x_vals = tuple(
        (i, j, k, l, n, t)
        for (i, j, k, l, n, t), var in x.items()
        if gurobi_on(var)
    )
    y_vals = tuple(
        (i, j, t)
        for (i, j, t), var in y.items()
        if gurobi_on(var)
    )
    return dict(obj=obj, optimal=(status == 2),
                time_out=(status in (9, 11)),
                n_vars=m.NumVars, n_constrs=m.NumConstrs,
                solve_time=m.Runtime, x_vals=x_vals, y_vals=y_vals)
