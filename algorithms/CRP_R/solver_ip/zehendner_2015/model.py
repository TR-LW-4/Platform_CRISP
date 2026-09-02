"""
BRP-II-A Gurobi formulation for Zehendner2015BRPIIA.

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
