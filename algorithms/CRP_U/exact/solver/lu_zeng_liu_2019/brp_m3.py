"""
BRP-m3 and BRP-m3R MIP formulations (Lu, Zeng & Liu 2019, Section IV).

Two models are implemented:
  BRP-m3   — full MIP that minimises the total number of relocations (§IV-A)
  BRP-m3R  — relaxed version that minimises remaining direct blockages
              after L relocations; used as a subroutine in IS* (§IV-B)

Variable naming follows the paper:
  x[t, i, j]   — block i directly upon block j at end of turn t  (B0 = B + {B+1})
  yhat[t, i, j] — block i lifted UP from block j in turn t
  y[t, i, j]   — block i placed DOWN upon block j in turn t
  z[t, i, j]   — block i retrieved from atop block j in turn t   (j > i only)
  u[t, i]      — height of block i after lift-down in turn t

Simplifications from Proposition 1 (always applied):
  (a) x[t,i,j] and u[t,i] are relaxed to continuous
  (b) constraints (yhat3), (z1), (z2) are eliminated
  (c) equality constraints (e1)-(e7) for block 1 are added

Reference
---------
C. Lu, B. Zeng, S. Liu,
"A Study on the Block Relocation Problem: Lower Bound Derivations and
Strong Formulations",
IEEE Transactions on Automation Science and Engineering, 2019.
arXiv:1904.03347
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional, Tuple


def _build_brp_m3(
    stacks: List[List[int]],
    n: int,
    W: int,
    H: int,
    T: int,
    L: int,
    time_limit_s: float,
    output_flag: int,
    is_relaxed: bool = False,
    adjacency: Optional[Dict[Tuple[int, int], int]] = None,
) -> Dict:
    """
    Build and solve BRP-m3 (or BRP-m3R if is_relaxed=True).

    Parameters
    ----------
    stacks        : list of stacks, each a list of priorities bottom→top
    n             : total number of blocks
    W             : number of stacks (unused directly; adjacency uses block pairs)
    H             : stack height limit
    T             : number of relocation turns to model
    L             : lower bound on number of relocations (used in constraint ^y1)
    time_limit_s  : Gurobi time limit
    output_flag   : Gurobi OutputFlag (0 = silent)
    is_relaxed    : if True build BRP-m3R (objective = remaining blockages after L turns)
    adjacency     : optional pre-computed C_{ij} dict; computed from stacks if None

    Returns dict with keys:
      obj          : int (optimal relocations) or None
      optimal      : bool
      time_out     : bool
      solve_time   : float
      n_vars       : int
      n_constrs    : int
      ops          : list of (from_block, to_block) relocation pairs (1-indexed) or None
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        print("[lu_zeng_liu_2019/brp_m3] ERROR: gurobipy not installed.", file=sys.stderr)
        return dict(obj=None, optimal=False, time_out=False, solve_time=0.0,
                    n_vars=0, n_constrs=0, ops=None)

    # ------------------------------------------------------------------ #
    # Build C_{ij} adjacency (block i directly above block j initially)  #
    # ------------------------------------------------------------------ #
    if adjacency is None:
        adjacency = {}
        for stk in stacks:
            for k in range(len(stk) - 1):
                j = stk[k]      # lower block
                i = stk[k + 1]  # upper block
                adjacency[(i, j)] = 1
            if stk:
                # topmost block rests on the floor (virtual block B+1)
                adjacency[(stk[-1], n + 1)] = 1

    # height of block i (1-indexed tier)
    height_of: Dict[int, int] = {}
    for stk in stacks:
        for h_idx, item in enumerate(stk, 1):
            height_of[item] = h_idx

    B_set = list(range(1, n + 1))    # blocks 1..n
    B0_set = B_set + [n + 1]          # extended set (n+1 = floor/virtual)
    ts_set = list(range(1, T + 1))    # turns 1..T
    ts0_set = [0] + ts_set            # turns 0..T

    # ------------------------------------------------------------------ #
    # Model                                                               #
    # ------------------------------------------------------------------ #
    m = gp.Model()
    m.Params.TimeLimit  = float(time_limit_s)
    m.Params.MIPGapAbs  = 0.99
    m.Params.OutputFlag = output_flag

    # --- Variables ---

    # x[t,i,j]: i directly upon j at end of turn t (continuous relaxation per Prop 1a)
    x_keys = [
        (t, i, j)
        for t in ts0_set
        for i in B_set
        for j in B0_set
        if i != j
    ]
    x = m.addVars(x_keys, lb=0.0, ub=1.0, name="x")

    # yhat[t,i,j]: lift-up of i from j in turn t (binary)
    yhat_keys = [(t, i, j) for t in ts_set for i in B_set for j in B0_set if i != j]
    yhat = m.addVars(yhat_keys, vtype=GRB.BINARY, name="yh")

    # y[t,i,j]: lift-down of i onto j in turn t (binary)
    y_keys = yhat_keys
    y = m.addVars(y_keys, vtype=GRB.BINARY, name="y")

    # z[t,i,j]: retrieval of i from atop j in turn t (j > i required, continuous)
    z_keys = [(t, i, j) for t in ts_set for i in B_set for j in B0_set if j > i]
    z = m.addVars(z_keys, lb=0.0, ub=1.0, name="z")

    # u[t,i]: height of block i after lift-down in turn t (continuous)
    u_keys = [(t, i) for t in ts_set for i in B_set]
    u = m.addVars(u_keys, lb=1.0, ub=float(H), name="u")

    # ------------------------------------------------------------------ #
    # Objective                                                           #
    # ------------------------------------------------------------------ #
    if is_relaxed:
        # BRP-m3R objective (o2): minimize direct blockages after L turns
        # = sum of x[L, i, j] where i > j (i is badly placed above j)
        obj_expr = gp.quicksum(
            x[L, i, j]
            for i in B_set for j in B_set
            if i != j and i > j and (L, i, j) in x
        )
        m.setObjective(obj_expr, GRB.MINIMIZE)
    else:
        # BRP-m3 objective (o1): minimize total lift-down operations
        obj_expr = gp.quicksum(y[t, i, j] for t, i, j in y_keys)
        m.setObjective(obj_expr, GRB.MINIMIZE)

    # ------------------------------------------------------------------ #
    # Constraints                                                         #
    # ------------------------------------------------------------------ #

    # (x1) initial configuration
    for i in B_set:
        for j in B0_set:
            if i == j:
                continue
            m.addConstr(x[0, i, j] == adjacency.get((i, j), 0), name=f"x1_{i}_{j}")

    # (x2) dynamic adjacency for j < i (no retrieval term)
    for t in ts_set:
        for i in B_set:
            for j in B0_set:
                if j >= i:
                    continue
                m.addConstr(
                    x[t, i, j] == x[t - 1, i, j] - yhat[t, i, j] + y[t, i, j],
                    name=f"x2_{t}_{i}_{j}",
                )

    # (x3) dynamic adjacency for j > i (includes retrieval term)
    for t in ts_set:
        for i in B_set:
            for j in B0_set:
                if j <= i:
                    continue
                m.addConstr(
                    x[t, i, j] == x[t - 1, i, j] - yhat[t, i, j] + y[t, i, j]
                    - (z[t, i, j] if (t, i, j) in z else 0),
                    name=f"x3_{t}_{i}_{j}",
                )

    # (x4) bay empty at end (only for full BRP-m3)
    if not is_relaxed:
        m.addConstr(
            gp.quicksum(x[T, i, j] for i in B_set for j in B0_set if i != j) == 0,
            name="x4",
        )

    # (^y1) exactly one lift-up per turn among first L turns
    for t in ts_set:
        if t <= L:
            m.addConstr(
                gp.quicksum(yhat[t, i, j] for i in B_set for j in B0_set if i != j) == 1,
                name=f"yh1_{t}",
            )
        else:
            m.addConstr(
                gp.quicksum(yhat[t, i, j] for i in B_set for j in B0_set if i != j) <= 1,
                name=f"yh1r_{t}",
            )

    # (^y2) monotone: if no lift-up at t, none afterwards
    for t in range(2, T + 1):
        m.addConstr(
            gp.quicksum(yhat[t, i, j] for i in B_set for j in B0_set if i != j) <=
            gp.quicksum(yhat[t - 1, i, j] for i in B_set for j in B0_set if i != j),
            name=f"yh2_{t}",
        )

    # (^y4) feasibility of lift-up: i must be the top of its stack
    for t in ts_set:
        for i in B_set:
            m.addConstr(
                gp.quicksum(yhat[t, i, j] for j in B0_set if i != j) <=
                gp.quicksum(x[t - 1, i, j] for j in B0_set if i != j) -
                gp.quicksum(x[t - 1, k, i] for k in B_set if k != i),
                name=f"yh4_{t}_{i}",
            )

    # (y1) same block lifted up and down per turn
    for t in ts_set:
        for i in B_set:
            m.addConstr(
                gp.quicksum(y[t, i, j] for j in B0_set if i != j) ==
                gp.quicksum(yhat[t, i, j] for j in B0_set if i != j),
                name=f"y1_{t}_{i}",
            )

    # (y2) lift-down only onto topmost block of destination stack
    for t in ts_set:
        for i in B_set:
            m.addConstr(
                gp.quicksum(x[t - 1, k, i] for k in B_set if k != i) >=
                gp.quicksum(y[t, k, i] for k in B_set if k != i),
                name=f"y2_{t}_{i}",
            )

    # (y3) lift-down onto floor only if stack empty
    for t in ts_set:
        m.addConstr(
            gp.quicksum(y[t, j, n + 1] for j in B_set) <=
            1 - gp.quicksum(yhat[t, j, n + 1] for j in B_set),
            name=f"y3_{t}",
        )

    # (y4) topmost constraint: i placed on top, so no block already on top
    for t in ts_set:
        for i in B_set:
            m.addConstr(
                gp.quicksum(x[t - 1, k, i] for k in B_set if k != i) -
                gp.quicksum(x[t - 1, i, k] for k in B_set if k != i) >=
                gp.quicksum(y[t, i, j] for j in B_set if i != j),
                name=f"y4_{t}_{i}",
            )

    # (y5) at most S-W empty floor slots used (limit drop to empty stacks)
    for t in ts_set:
        floor_slots = sum(
            1 for stk in stacks if not stk
        )  # initial empty stacks; approximation
        m.addConstr(
            gp.quicksum(y[t, j, n + 1] for j in B_set) <=
            W - gp.quicksum(x[t - 1, k, n + 1] for k in B_set),
            name=f"y5_{t}",
        )

    # (z3) retrieval order: block i retrieved before block i-1 impossible
    for t in ts_set:
        for i in range(2, n + 1):
            m.addConstr(
                gp.quicksum(
                    z[tt, i, j]
                    for tt in range(1, t + 1)
                    for j in B0_set
                    if j > i and (tt, i, j) in z
                ) <=
                gp.quicksum(
                    z[tt, i - 1, j]
                    for tt in range(1, t + 1)
                    for j in B0_set
                    if j > i - 1 and (tt, i - 1, j) in z
                ),
                name=f"z3_{t}_{i}",
            )

    # (u1) height consistency
    for t in ts_set:
        for i in B_set:
            for j in B_set:
                if i == j:
                    continue
                m.addConstr(
                    u[t, i] >= u[t, j] + 1 - H * (
                        1 - x[t - 1, i, j] + yhat[t, i, j] - y[t, i, j]
                    ),
                    name=f"u1_{t}_{i}_{j}",
                )

    # (e1)-(e7) block 1 constraints (Proposition 1c)
    # Find block 1 and its initial lower block j1
    j1 = None
    for stk in stacks:
        for h_idx, item in enumerate(stk):
            if item == 1:
                j1 = stk[h_idx - 1] if h_idx > 0 else n + 1
                break
        if j1 is not None:
            break

    if j1 is not None:
        # (e1) block 1 retrieved when j1 is lifted up
        for t in ts_set:
            m.addConstr(
                gp.quicksum(z[t, 1, j] for j in B0_set if j > 1 and (t, 1, j) in z) ==
                yhat[t, j1, 1] if (t, j1, 1) in yhat else 0,
                name=f"e1_{t}",
            )

        # (e2) j1 lifted up exactly once from block 1
        m.addConstr(
            gp.quicksum(yhat[t, j1, 1] for t in ts_set if (t, j1, 1) in yhat) == 1,
            name="e2",
        )

        # (e3)-(e7) block 1 not relocatable after retrieval (simplified)
        for t in ts_set:
            m.addConstr(
                gp.quicksum(y[t, k, 1] for k in B_set if k != 1) == 0,
                name=f"e3_{t}",
            )
            # (e7) height of block 1 = initial height
        h1 = height_of.get(1, 1)
        for t in ts_set:
            m.addConstr(u[t, 1] == h1, name=f"e7_{t}")

    m.optimize()

    status = m.Status
    feasible = m.SolCount > 0

    if not feasible:
        return dict(
            obj=None, optimal=False,
            time_out=(status in (9, 11)),
            solve_time=m.Runtime,
            n_vars=m.NumVars, n_constrs=m.NumConstrs,
            ops=None,
        )

    obj_val = int(round(m.ObjVal)) if not is_relaxed else int(round(m.ObjVal))

    # Extract solution (relocation plan) from y variables
    ops: List[Tuple[int, int]] = []
    if not is_relaxed:
        for t in ts_set:
            for i in B_set:
                for j in B0_set:
                    if i != j and (t, i, j) in y and y[t, i, j].X > 0.5:
                        ops.append((i, j))   # block i placed upon block j
            # also record retrievals
            for i in B_set:
                for j in B0_set:
                    if j > i and (t, i, j) in z and z[t, i, j].X > 0.5:
                        ops.append((i, -(j)))  # negative j = retrieval marker

    return dict(
        obj=obj_val,
        optimal=(status == GRB.OPTIMAL),
        time_out=(status in (9, 11)),
        solve_time=m.Runtime,
        n_vars=m.NumVars,
        n_constrs=m.NumConstrs,
        ops=ops if ops else None,
    )


def solve_brp_m3(
    stacks: List[List[int]],
    n: int,
    W: int,
    H: int,
    L: int,
    T: int,
    time_limit_s: float = 3600.0,
    output_flag: int = 0,
) -> Dict:
    """
    Solve BRP-m3 (full MIP).

    Parameters
    ----------
    stacks        : initial stacks (bottom→top, 1-indexed priorities)
    n, W, H       : blocks, stacks, height limit
    L             : lower bound used in (^y1) constraint
    T             : number of turns to model (upper bound)
    """
    return _build_brp_m3(
        stacks, n, W, H, T, L,
        time_limit_s, output_flag,
        is_relaxed=False,
    )


def solve_brp_m3r(
    stacks: List[List[int]],
    n: int,
    W: int,
    H: int,
    L: int,
    time_limit_s: float = 3600.0,
    output_flag: int = 0,
) -> Dict:
    """
    Solve BRP-m3R (relaxed, IS* subroutine).

    T = L in BRP-m3R (model only L turns, minimise remaining blockages).
    """
    return _build_brp_m3(
        stacks, n, W, H,
        T=L, L=L,
        time_limit_s=time_limit_s,
        output_flag=output_flag,
        is_relaxed=True,
    )


# ------------------------------------------------------------------ #
# Self-contained MinMax upper bound (used to initialise T)           #
# ------------------------------------------------------------------ #

def minmax_ub(stacks: List[List[int]], n: int, W: int, H: int) -> int:
    """
    Caserta (2012) MinMax upper bound — self-contained, no platform deps.
    """
    import copy
    stks = [list(s) for s in stacks]
    heights = [len(s) for s in stks]

    def low_of(s: int) -> int:
        return min(stks[s]) if stks[s] else n + 1

    lows = [low_of(s) for s in range(W)]
    n_reloc = 0

    for tp in range(1, n + 1):
        ts = ti = None
        for si in range(W):
            for i, p in enumerate(stks[si]):
                if p == tp:
                    ts, ti = si, i
                    break
            if ts is not None:
                break
        if ts is None:
            continue
        while len(stks[ts]) > ti + 1:
            n_reloc += 1
            r = stks[ts][-1]
            good = [s for s in range(W) if s != ts and heights[s] < H and lows[s] > r]
            if good:
                dst = min(good, key=lambda s: lows[s])
            else:
                avail = [s for s in range(W) if s != ts and heights[s] < H]
                if not avail:
                    break
                dst = max(avail, key=lambda s: lows[s])
            stks[ts].pop()
            heights[ts] -= 1
            stks[dst].append(r)
            heights[dst] += 1
            lows[dst] = low_of(dst)
            lows[ts]  = low_of(ts)
        stks[ts] = [p for p in stks[ts] if p != tp]
        heights[ts] -= 1
        if stks[ts]:
            lows[ts] = low_of(ts)

    return n_reloc
