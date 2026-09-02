"""
ILP / ILP-DK Gurobi formulation for Tang2015ILP.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from core.plan import Movement, RelocationPlan
from core.yard import Yard
from algorithms.CRP_R.solver_ip.common import (
    occupancy_stages_to_plan,
    priority_to_container_id,
    stack_keys,
)


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


def _read_occupancy(get_x, stage: int, i_from: int, N: int, C: int, P: int) -> Dict[int, Tuple[int, int]]:
    pos: Dict[int, Tuple[int, int]] = {}
    for i in range(i_from, N + 1):
        found = False
        for c in range(1, C + 1):
            for p in range(1, P + 1):
                val = get_x(stage, i, c, p)
                raw = val.X if hasattr(val, "X") else val
                if float(raw) > 0.5:
                    pos[i] = (c, p)
                    found = True
                    break
            if found:
                break
    return pos


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
    occupancy: Dict[int, Dict[int, Tuple[int, int]]] = {1: dict(initial_pos)}
    for stage in range(2, n_stages + 2):
        if stage > N:
            break
        occupancy[stage] = _read_occupancy(get_x, stage, stage, N, C, P)
    x_next = dict(occupancy.get(2) or {})

    return dict(
        obj=obj_val,
        optimal=(status == GRB.OPTIMAL),
        time_out=(status in (9, 11)),
        n_vars=m.NumVars,
        n_constrs=m.NumConstrs,
        solve_time=m.Runtime,
        y_vals=y_vals,
        x_next=x_next,
        occupancy=occupancy,
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
                    n_vars=0, n_constrs=0, solve_time=0.0, plan=None)
    keys = stack_keys(yard)
    pri = priority_to_container_id(yard)
    result["plan"] = occupancy_stages_to_plan(
        result.get("occupancy") or {1: initial_pos},
        result.get("y_vals") or {},
        N,
        keys,
        pri,
        max_tiers=H,
    )
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
    keys = stack_keys(yard)
    pri_to_id = priority_to_container_id(yard)
    plan = RelocationPlan()

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

        target_g = remaining[0]
        if result is None:
            tc, tp = current_pos[target_g]
            for g in remaining[1:]:
                gc, gp_ = current_pos[g]
                if gc == tc and gp_ > tp:
                    total_reloc += 1
        else:
            y_vals = result["y_vals"]
            movers = [
                li for li in range(2, n_rem + 1)
                if float(y_vals.get((1, li), 0) or 0) > 0.5
            ]
            movers.sort(key=lambda li: -int(local_pos[li][1]))
            heights = [0] * C
            for _g, (c, p) in current_pos.items():
                heights[c - 1] = max(heights[c - 1], p)
            target_c, _tp = current_pos[target_g]
            for li in movers:
                g = local_to_global[li]
                src_c, _sp = current_pos[g]
                nxt = result["x_next"].get(li)
                if nxt is not None and nxt[0] != src_c:
                    dst_c = nxt[0]
                else:
                    dst_c = next(
                        (
                            c for c in range(1, C + 1)
                            if c != src_c and c != target_c and heights[c - 1] < H
                        ),
                        next((c for c in range(1, C + 1) if c != src_c), src_c),
                    )
                plan.add(Movement(
                    int(pri_to_id[g]),
                    keys[src_c - 1],
                    keys[dst_c - 1],
                ))
                heights[src_c - 1] -= 1
                heights[dst_c - 1] += 1
                current_pos[g] = (dst_c, heights[dst_c - 1])
            total_reloc += result["obj"] if K_eff == n_rem - 1 else len(movers)
            all_vars    += result["n_vars"]
            all_constrs += result["n_constrs"]

        src_c, _sp = current_pos[target_g]
        plan.add(Movement(int(pri_to_id[target_g]), keys[src_c - 1], None))

        retrieved = remaining.pop(0)
        del current_pos[retrieved]

    if remaining:
        last = remaining[0]
        src_c, _sp = current_pos[last]
        plan.add(Movement(int(pri_to_id[last]), keys[src_c - 1], None))

    return dict(
        obj=total_reloc,
        optimal=False,
        time_out=(total_time >= time_limit_s),
        n_vars=all_vars,
        n_constrs=all_constrs,
        solve_time=total_time,
        plan=plan,
    )
