"""
BRP-III Gurobi formulation for PeteringHussein2013BRPIII.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple


Move = Tuple[int, int, Optional[int]]


def solve_brp_iii(
    stacks: Sequence[Sequence[int]],
    max_height: int,
    horizon: int,
    time_limit_s: float = 3600.0,
    output_flag: int = 0,
    enforce_adjacent_height: bool = False,
) -> Dict:
    """
    Build and solve BRP-III.

    Parameters
    ----------
    stacks
        Initial stacks, bottom-to-top, with unique priorities 1..C.
    max_height
        Maximum number of containers in any stack.
    horizon
        Maximum total moves W (retrievals plus relocations).
    enforce_adjacent_height
        Enable optional safety constraints (7)-(8). The paper disables
        these constraints in its BRP-I/BRP-III computational comparison.

    Returns
    -------
    A dictionary containing the objective, relocation count, solve status,
    model size, and ``moves`` as ``(priority, src, dst_or_None)`` tuples.
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return {
            "obj": None,
            "relocations": None,
            "total_moves": None,
            "optimal": False,
            "time_out": False,
            "solve_time": 0.0,
            "n_vars": 0,
            "n_constrs": 0,
            "moves": None,
            "error": "gurobipy is not installed",
        }

    initial = [list(stack) for stack in stacks]
    C = sum(len(stack) for stack in initial)
    S = len(initial)
    H = int(max_height)
    W = int(horizon)

    if C == 0:
        return {
            "obj": 0,
            "relocations": 0,
            "total_moves": 0,
            "optimal": True,
            "time_out": False,
            "solve_time": 0.0,
            "n_vars": 0,
            "n_constrs": 0,
            "moves": [],
            "error": None,
        }
    priorities = sorted(p for stack in initial for p in stack)
    if priorities != list(range(1, C + 1)):
        raise ValueError("BRP-III requires unique consecutive priorities 1..C")
    if H < max(len(stack) for stack in initial):
        raise ValueError("max_height is below an initial stack height")
    if W < C:
        raise ValueError("horizon must include at least C retrieval moves")

    cs = range(1, C + 1)
    ss = range(1, S + 1)
    ts = range(1, W + 1)
    state_ts = range(1, W + 2)

    initial_setup = {
        (c, s): 0
        for c in cs
        for s in ss
    }
    initial_bury = {c: 0 for c in cs}
    for s, stack in enumerate(initial, 1):
        for level, c in enumerate(stack):
            initial_setup[c, s] = 1
            initial_bury[c] = len(stack) - level

    model = gp.Model("Petering_Hussein_2013_BRP_III")
    model.Params.TimeLimit = float(time_limit_s)
    model.Params.MIPGapAbs = 0.99
    model.Params.OutputFlag = int(output_flag)

    # Table 3 decision variables. Four three-index variable families are
    # continuous by design; constraints (1)-(3) force binary values.
    X = model.addVars(cs, ss, state_ts, lb=0.0, ub=1.0, name="X")
    B = model.addVars(cs, state_ts, lb=0.0, ub=float(H), name="B")
    M = model.addVars(cs, ts, vtype=GRB.BINARY, name="M")
    Closer = model.addVars(cs, ts, vtype=GRB.BINARY, name="C")
    Farther = model.addVars(cs, ts, vtype=GRB.BINARY, name="F")
    Taken = model.addVars(cs, ts, vtype=GRB.BINARY, name="T")
    Removed = model.addVars(ss, ts, vtype=GRB.BINARY, name="R")
    Placed = model.addVars(ss, ts, vtype=GRB.BINARY, name="P")
    Rc = model.addVars(cs, ss, ts, lb=0.0, ub=1.0, name="Rc")
    Pc = model.addVars(cs, ss, ts, lb=0.0, ub=1.0, name="Pc")

    model.setObjective(
        gp.quicksum(t * Taken[C, t] for t in ts),
        GRB.MINIMIZE,
    )

    # (1a)-(1f): Rc = Removed AND M.
    for c in cs:
        for s in ss:
            for t in ts:
                model.addConstr(Rc[c, s, t] <= Removed[s, t])
                model.addConstr(Rc[c, s, t] <= M[c, t])
                model.addConstr(Rc[c, s, t] >= Removed[s, t] + M[c, t] - 1)
    for s in ss:
        for t in ts:
            model.addConstr(gp.quicksum(Rc[c, s, t] for c in cs) == Removed[s, t])
    for c in cs:
        for t in ts:
            model.addConstr(gp.quicksum(Rc[c, s, t] for s in ss) == M[c, t])

    # (2a)-(2f): Pc = Placed AND M; retrieval moves have no placement.
    for c in cs:
        for s in ss:
            for t in ts:
                model.addConstr(Pc[c, s, t] <= Placed[s, t])
                model.addConstr(Pc[c, s, t] <= M[c, t])
                model.addConstr(Pc[c, s, t] >= Placed[s, t] + M[c, t] - 1)
    for s in ss:
        for t in ts:
            model.addConstr(gp.quicksum(Pc[c, s, t] for c in cs) == Placed[s, t])
    for c in cs:
        for t in ts:
            model.addConstr(gp.quicksum(Pc[c, s, t] for s in ss) <= M[c, t])

    # (3a)-(3c): stack membership state.
    for c in cs:
        for s in ss:
            model.addConstr(X[c, s, 1] == initial_setup[c, s])
            for t in ts:
                model.addConstr(
                    X[c, s, t + 1] == X[c, s, t] + Pc[c, s, t] - Rc[c, s, t]
                )

    # (4a)-(4c): number of containers burying c, including c itself.
    for c in cs:
        model.addConstr(B[c, 1] == initial_bury[c])
        for t in ts:
            model.addConstr(
                B[c, t + 1] == B[c, t] + Farther[c, t] - Closer[c, t]
            )

    # (5)-(6): membership and stack capacity.
    for c in cs:
        for t in state_ts:
            model.addConstr(gp.quicksum(X[c, s, t] for s in ss) <= 1)
    for s in ss:
        for t in state_ts:
            model.addConstr(gp.quicksum(X[c, s, t] for c in cs) <= H)

    # Optional safety constraints (7)-(8).
    if enforce_adjacent_height:
        for s in range(1, S):
            for t in state_ts:
                lhs = gp.quicksum(X[c, s, t] for c in cs)
                rhs = gp.quicksum(X[c, s + 1, t] for c in cs)
                model.addConstr(lhs <= rhs + 2)
                model.addConstr(lhs >= rhs - 2)

    # (9)-(15): move and burial accounting.
    for c in cs:
        model.addConstr(gp.quicksum(M[c, t] for t in ts) >= 1)
        model.addConstr(
            gp.quicksum(Closer[c, t] - Farther[c, t] for t in ts)
            == initial_bury[c]
        )
    for t in ts:
        model.addConstr(gp.quicksum(M[c, t] for c in cs) <= 1)
        model.addConstr(gp.quicksum(Closer[c, t] for c in cs) <= H)
        model.addConstr(gp.quicksum(Farther[c, t] for c in cs) <= H)
    for t in range(1, C + 1):
        model.addConstr(gp.quicksum(M[c, t] for c in cs) == 1)

    # (16)-(18): each container is retrieved once and in priority order.
    for t in ts:
        model.addConstr(gp.quicksum(Taken[c, t] for c in cs) <= 1)
    for c in cs:
        model.addConstr(gp.quicksum(Taken[c, t] for t in ts) == 1)
    for c in range(1, C):
        model.addConstr(
            gp.quicksum(t * Taken[c + 1, t] for t in ts)
            >= 1 + gp.quicksum(t * Taken[c, t] for t in ts)
        )

    # (19)-(23): removal/placement restrictions.
    for t in ts:
        model.addConstr(gp.quicksum(Removed[s, t] for s in ss) <= 1)
        model.addConstr(
            gp.quicksum(Placed[s, t] for s in ss)
            <= gp.quicksum(Removed[s, t] for s in ss)
        )
        for s in ss:
            model.addConstr(Placed[s, t] + Removed[s, t] <= 1)
    for t in range(1, C + 1):
        model.addConstr(gp.quicksum(Removed[s, t] for s in ss) == 1)
    for c in cs:
        for t in ts:
            model.addConstr(
                gp.quicksum(Pc[c, s, t] for s in ss)
                <= gp.quicksum(Rc[c, s, t] for s in ss)
            )

    # (24)-(27): top accessibility and retrieval identity.
    for c in cs:
        for s in ss:
            for t in ts:
                model.addConstr(Rc[c, s, t] <= X[c, s, t])
        for t in ts:
            model.addConstr(
                B[c, t] - 1 <= (1 - M[c, t]) * (H - 1)
            )
            model.addConstr(Taken[c, t] <= M[c, t])
            model.addConstr(
                Taken[c, t]
                == gp.quicksum(Rc[c, s, t] - Pc[c, s, t] for s in ss)
            )

    # (28a)-(28c): Closer = X(c,s,t) AND Removed(s,t).
    for c in cs:
        for s in ss:
            for t in ts:
                model.addConstr(X[c, s, t] >= Closer[c, t] + Removed[s, t] - 1)
                model.addConstr(Removed[s, t] >= X[c, s, t] + Closer[c, t] - 1)
                model.addConstr(Closer[c, t] >= Removed[s, t] + X[c, s, t] - 1)

    # (29a)-(29c): Farther = X(c,s,t+1) AND Placed(s,t).
    for c in cs:
        for s in ss:
            for t in ts:
                model.addConstr(
                    X[c, s, t + 1] >= Farther[c, t] + Placed[s, t] - 1
                )
                model.addConstr(
                    Placed[s, t] >= X[c, s, t + 1] + Farther[c, t] - 1
                )
                model.addConstr(
                    Farther[c, t] >= Placed[s, t] + X[c, s, t + 1] - 1
                )

    model.optimize()

    status = model.Status
    timeout_statuses = {
        GRB.TIME_LIMIT,
        GRB.INTERRUPTED,
        getattr(GRB, "MEM_LIMIT", -999),
    }
    if model.SolCount == 0:
        return {
            "obj": None,
            "relocations": None,
            "total_moves": None,
            "optimal": False,
            "time_out": status in timeout_statuses,
            "solve_time": model.Runtime,
            "n_vars": model.NumVars,
            "n_constrs": model.NumConstrs,
            "moves": None,
            "error": None,
        }

    moves: List[Move] = []
    for t in ts:
        moved = [c for c in cs if M[c, t].X > 0.5]
        if not moved:
            continue
        c = moved[0]
        srcs = [s for s in ss if Removed[s, t].X > 0.5]
        if not srcs:
            continue
        dsts = [s for s in ss if Placed[s, t].X > 0.5]
        dst = dsts[0] - 1 if dsts else None
        moves.append((c, srcs[0] - 1, dst))

    relocations = sum(1 for _, _, dst in moves if dst is not None)
    return {
        "obj": int(round(model.ObjVal)),
        "relocations": relocations,
        "total_moves": len(moves),
        "optimal": status == GRB.OPTIMAL,
        "time_out": status in timeout_statuses,
        "solve_time": model.Runtime,
        "n_vars": model.NumVars,
        "n_constrs": model.NumConstrs,
        "moves": moves,
        "error": None,
    }
