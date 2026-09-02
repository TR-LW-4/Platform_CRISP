"""
CRP-I Gurobi formulation for GalleCRPI2018.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

from core.yard import Yard
from algorithms.CRP_R.solver_ip.common import (
    gurobi_on,
)


# ================================================================ #
#  Binary encoding helpers                                           #
# ================================================================ #

def _yard_to_binary_encoding(yard: Yard, C: int, S: int) -> List[List[int]]:
    """
    Build the initial binary encoding A (1-indexed).

    Convention (Galle 2017):
      A[c][d] = 1  iff container with priority c is physically below
                    container with priority d (same stack).
      A[C+s][d] = 1  iff container d is in stack s  (s = 1 .. S).

    Row 0 and column 0 are unused (1-indexed padding).
    """
    A = [[0] * (C + 2) for _ in range(C + S + 2)]
    stack_list = sorted(yard.stacks.values(), key=lambda s: (s.bay, s.row))
    for s_idx, stack in enumerate(stack_list, 1):
        containers = stack.containers  # index 0 = physically bottom
        for i, cont in enumerate(containers):
            c = cont.priority  # 1 .. C
            # Artificial sentinel: container d is in stack s_idx
            A[C + s_idx][c] = 1
            # All containers at higher indices are physically above cont
            for j in range(i + 1, len(containers)):
                A[c][containers[j].priority] = 1  # c is below containers[j]
    return A


def _build_encoding_from_priority_stacks(
    stacks: List[List[int]], C: int, S: int
) -> List[List[int]]:
    """Same as _yard_to_binary_encoding but from a list of priority lists."""
    A = [[0] * (C + 2) for _ in range(C + S + 2)]
    for s_idx, stack in enumerate(stacks, 1):
        for i, c in enumerate(stack):
            A[C + s_idx][c] = 1
            for j in range(i + 1, len(stack)):
                A[c][stack[j]] = 1
    return A


def _compute_first_move(A_init: List[List[int]], C: int) -> List[int]:
    """
    firstMove[c] = first period n in which container c is moved.

    c is moved in period n when:
      n == c  (c is the target and gets retrieved), OR
      A[n][c] == 1  (c is physically above target n → c must be relocated).
    """
    first_move = [0] * (C + 1)  # 1-indexed; index 0 unused
    for c in range(1, C + 1):
        for n in range(1, C + 1):
            if n == c or A_init[n][c] == 1:
                first_move[c] = n
                break
    return first_move


def _count_a_variables(C: int, S: int) -> int:
    """Count the number of 'a' decision variables (for memory pre-check)."""
    N = max(1, C - S + 1)
    return sum((C + S - n + 1) * (C - n + 1) for n in range(1, N + 1))


def _occupancy_from_galle_a(
    a_on: set,
    n: int,
    C: int,
    S: int,
) -> Dict[int, Tuple[int, int]]:
    """Decode {priority: (col, tier)} from CRP-I a[n,c,d]=1 tuples."""
    pos: Dict[int, Tuple[int, int]] = {}
    for d in range(n, C + 1):
        col = None
        below = 0
        for s in range(1, S + 1):
            if (n, C + s, d) in a_on:
                col = s
                break
        for c in range(n, C + 1):
            if c != d and (n, c, d) in a_on:
                below += 1
        if col is not None:
            pos[d] = (col, below + 1)
    return pos


# ================================================================ #
#  MinMax heuristic (Caserta 2012) – upper bound & warm start        #
# ================================================================ #

def _run_minmax_heuristic(
    stacks_init: List[List[int]],
    C: int,
    S: int,
    T: int,
) -> Tuple[int, Dict[int, List[List[int]]]]:
    """
    Caserta et al. (2012) MinMax heuristic.

    Returns
    -------
    n_reloc : number of relocations performed.
    history : dict mapping period n → priority stacks at the START of period n
              (before any relocations for that period).
    """
    stacks = [list(s) for s in stacks_init]
    height = [len(s) for s in stacks]

    def _min_pri(s_idx: int) -> int:
        return min(stacks[s_idx]) if stacks[s_idx] else C + 1

    min_pri = [_min_pri(s) for s in range(S)]
    n_reloc = 0
    history: Dict[int, List[List[int]]] = {}

    for target_p in range(1, C + 1):
        history[target_p] = [list(s) for s in stacks]

        # Locate target in stacks
        target_s = target_pos = None
        for s in range(S):
            for i, p in enumerate(stacks[s]):
                if p == target_p:
                    target_s, target_pos = s, i
                    break
            if target_s is not None:
                break
        if target_s is None:
            continue

        # Relocate containers above the target (LIFO: from top)
        while len(stacks[target_s]) > target_pos + 1:
            n_reloc += 1
            r = stacks[target_s][-1]

            good = [s for s in range(S)
                    if s != target_s and height[s] < T and min_pri[s] > r]
            if good:
                dst = min(good, key=lambda s: min_pri[s])
            else:
                avail = [s for s in range(S)
                         if s != target_s and height[s] < T]
                if not avail:
                    break
                dst = max(avail, key=lambda s: min_pri[s])

            stacks[target_s].pop()
            height[target_s] -= 1
            stacks[dst].append(r)
            height[dst] += 1
            min_pri[dst] = _min_pri(dst)
            min_pri[target_s] = _min_pri(target_s)

        # Retrieve the target
        stacks[target_s] = [p for p in stacks[target_s] if p != target_p]
        height[target_s] -= 1
        min_pri[target_s] = _min_pri(target_s)

    return n_reloc, history


# ================================================================ #
#  Core Gurobi IP solver                                             #
# ================================================================ #

def _solve_ip(
    yard: Yard,
    C: int,
    S: int,
    T: int,
    time_limit_s: float,
    use_gap: bool,
    use_incumbent: bool,
    use_preprocessing: bool,
    memory_limit: float,
    output_flag: int,
) -> Dict:
    """
    Build and solve the CRP-I binary IP.

    Returns a result dict with keys:
      obj, optimal, time_out, memory_out, n_vars, n_constrs, solve_time
    """
    import gurobipy as gp
    from gurobipy import GRB

    N = C - S + 1  # number of retrieval periods tracked explicitly

    # ── Trivial / edge cases ────────────────────────────────────────── #
    if C < S:
        # More stacks than containers → always solvable with 0 relocations
        return dict(obj=0, optimal=True, time_out=False,
                    memory_out=False, n_vars=0, n_constrs=0, solve_time=0.0,
                    occupancy=None, y_vals={})

    if C == S:
        # N = 1: only b variables; objective = number of blocking containers
        A = _yard_to_binary_encoding(yard, C, S)
        blocking = sum(
            1 for d in range(2, C + 1)
            if any(A[c][d] == 1 for c in range(1, d))
        )
        return dict(obj=blocking, optimal=True, time_out=False,
                    memory_out=False, n_vars=C - 1, n_constrs=0, solve_time=0.0,
                    occupancy=None, y_vals={})

    # ── Quick memory pre-check (variable count) ─────────────────────── #
    n_vars_approx = _count_a_variables(C, S)
    # Rough constraint estimate: ~3–4× variable count
    if n_vars_approx * (n_vars_approx * 4) >= memory_limit:
        return dict(obj=None, optimal=False, time_out=False,
                    memory_out=True, n_vars=n_vars_approx, n_constrs=0,
                    solve_time=0.0, occupancy=None, y_vals={})

    # ── Build binary encoding ───────────────────────────────────────── #
    A_init = _yard_to_binary_encoding(yard, C, S)

    # ── Build Gurobi model ──────────────────────────────────────────── #
    mip_gap_abs = 0.99 if use_gap else 0.0
    m = gp.Model()
    m.Params.TimeLimit  = float(time_limit_s)
    m.Params.MIPGapAbs  = mip_gap_abs
    m.Params.OutputFlag = output_flag

    # ── Variables ───────────────────────────────────────────────────── #
    # a[n, c, d]: 1-indexed, n=1..N, c=n..C+S, d=n..C
    a: Dict[Tuple[int,int,int], gp.Var] = {}
    for n in range(1, N + 1):
        for c in range(n, C + S + 1):
            for d in range(n, C + 1):
                a[n, c, d] = m.addVar(vtype=GRB.BINARY)

    # b[d]: 1-indexed, d=N+1..C
    b: Dict[int, gp.Var] = {}
    for d in range(N + 1, C + 1):
        b[d] = m.addVar(vtype=GRB.BINARY)

    m.update()

    # ── Memory check with exact counts ──────────────────────────────── #
    n_vars = m.NumVars

    # ── Objective ───────────────────────────────────────────────────── #
    obj_expr = gp.quicksum(
        a[n, n, d] for n in range(1, N) for d in range(n + 1, C + 1)
    )
    if b:
        obj_expr = obj_expr + gp.quicksum(b.values())
    m.setObjective(obj_expr, GRB.MINIMIZE)

    # ── Constraint 1: initialise from binary encoding ────────────────── #
    for c in range(1, C + S + 1):
        for d in range(1, C + 1):
            m.addConstr(a[1, c, d] == A_init[c][d])

    # ── Constraint 2: each container in exactly one stack ─────────────── #
    for n in range(2, N + 1):
        for c in range(n, C + 1):
            m.addConstr(
                gp.quicksum(a[n, C + s, c] for s in range(1, S + 1)) == 1
            )

    # ── Constraint 3: no self-below ─────────────────────────────────── #
    for n in range(2, N + 1):
        for c in range(n, C + 1):
            m.addConstr(a[n, c, c] == 0)

    # ── Constraint 4: antisymmetry ──────────────────────────────────── #
    for n in range(2, N + 1):
        for c in range(n, C + 1):
            for d in range(c + 1, C + 1):  # d > c avoids duplicates
                m.addConstr(a[n, c, d] + a[n, d, c] <= 1)

    # ── Constraints 5 & 6: same-stack / different-stack ordering ────── #
    for n in range(2, N + 1):
        for s in range(1, S + 1):
            for c in range(n, C + 1):
                for d in range(n, C + 1):
                    if d == c:
                        continue
                    # C5: if c and d are in the same stack, one must be below the other
                    m.addConstr(
                        a[n, c, d] + a[n, d, c]
                        - a[n, C + s, c] - a[n, C + s, d] >= -1
                    )
                    # C6: if c and d are in different stacks, neither is below the other
                    m.addConstr(
                        a[n, c, d] + a[n, d, c] + a[n, C + s, c]
                        + gp.quicksum(
                            a[n, C + r, d] for r in range(1, S + 1) if r != s
                        ) <= 2
                    )

    # ── Constraint 7: stack capacity ────────────────────────────────── #
    for n in range(2, N + 1):
        for s in range(1, S + 1):
            m.addConstr(
                gp.quicksum(a[n, C + s, d] for d in range(n, C + 1)) <= T
            )

    # ── Constraint 8: b variables (blocking after N-1 retrievals) ───── #
    for d in range(N + 1, C + 1):
        for c in range(N, d):
            m.addConstr(b[d] - a[N, c, d] >= 0)

    # ── Constraints 9 & 10: column evolution between periods ────────── #
    for n in range(1, N):
        for c in range(n + 1, C + 1):
            for d in range(n + 1, C + S + 1):
                if d == c:
                    continue
                m.addConstr(a[n + 1, d, c] - a[n, d, c] - a[n, n, c] <= 0)
                m.addConstr(a[n + 1, d, c] - a[n, d, c] + a[n, n, c] >= 0)

    # ── Constraint 11: relocated container leaves its stack ─────────── #
    for n in range(1, N):
        for c in range(n + 1, C + 1):
            for s in range(1, S + 1):
                m.addConstr(
                    a[n, n, c] + a[n, C + s, c] + a[n + 1, C + s, c] <= 2
                )

    # ── Constraint 12: LIFO ordering ────────────────────────────────── #
    for n in range(1, N):
        for c in range(n + 1, C + 1):
            for d in range(n + 1, C + 1):
                if d == c:
                    continue
                m.addConstr(
                    a[n, n, c] + a[n, n, d] + a[n, c, d] + a[n + 1, c, d] <= 3
                )

    m.update()
    n_constrs = m.NumConstrs

    # ── Exact memory check ───────────────────────────────────────────── #
    if n_vars * n_constrs >= memory_limit:
        m.dispose()
        return dict(obj=None, optimal=False, time_out=False,
                    memory_out=True, n_vars=n_vars, n_constrs=n_constrs,
                    solve_time=0.0, occupancy=None, y_vals={})

    # ── Pre-processing: fix columns that don't change yet ─────────────── #
    if use_preprocessing:
        first_move = _compute_first_move(A_init, C)
        for c in range(1, C + 1):
            # Column c of binary encoding is unchanged for periods 2 .. firstMove[c]
            upper = min(first_move[c], N)
            for n in range(2, upper + 1):
                for d in range(n, C + S + 1):
                    m.addConstr(a[n, d, c] == A_init[d][c])

    # ── Incumbent: warm-start from MinMax heuristic ─────────────────── #
    if use_incumbent:
        stacks_init = [
            [cont.priority for cont in stack.containers]
            for stack in sorted(
                yard.stacks.values(), key=lambda s: (s.bay, s.row)
            )
        ]
        _n_heur, history = _run_minmax_heuristic(stacks_init, C, S, T)

        for n in range(1, N + 1):
            A_n = _build_encoding_from_priority_stacks(
                history.get(n, stacks_init), C, S
            )
            for c in range(n, C + S + 1):
                for d in range(n, C + 1):
                    if (n, c, d) in a:
                        a[n, c, d].Start = float(A_n[c][d])

        # b warm-start from the terminal configuration A^(N)
        A_N = _build_encoding_from_priority_stacks(
            history.get(N, stacks_init), C, S
        )
        for d in range(N + 1, C + 1):
            if d in b:
                b[d].Start = float(
                    1 if any(A_N[c][d] == 1 for c in range(N, d)) else 0
                )

    # ── Solve ────────────────────────────────────────────────────────── #
    t0 = time.perf_counter()
    m.optimize()
    solve_time = time.perf_counter() - t0

    status = m.Status
    if m.SolCount == 0:
        obj_val = None
        optimal = False
    else:
        obj_val = int(round(m.ObjVal))
        optimal = (status == GRB.OPTIMAL)

    time_out = status in (GRB.TIME_LIMIT, GRB.INTERRUPTED)

    occupancy = None
    y_vals: Dict[Tuple[int, int], int] = {}
    if obj_val is not None:
        a_on = {
            (n, c, d) for (n, c, d), var in a.items() if gurobi_on(var)
        }
        occupancy = {
            n: _occupancy_from_galle_a(a_on, n, C, S)
            for n in range(1, N + 1)
        }
        for n in range(1, N):
            for d in range(n + 1, C + 1):
                if (n, n, d) in a_on:
                    y_vals[n, d] = 1

    return dict(
        obj=obj_val,
        optimal=optimal,
        time_out=time_out,
        memory_out=False,
        n_vars=n_vars,
        n_constrs=n_constrs,
        solve_time=solve_time,
        occupancy=occupancy,
        y_vals=y_vals,
    )
