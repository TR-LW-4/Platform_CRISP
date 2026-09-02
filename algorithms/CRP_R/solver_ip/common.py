"""
Shared helpers for CRP-R IP wrappers: layout maps, plan JSON, metric merge.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from core.plan import Movement, RelocationPlan
from core.yard import Yard


def stack_keys(yard: Yard) -> List[Tuple[int, int]]:
    """1-indexed stack ``i`` in IP models is ``stack_keys[i - 1]`` = (bay, row)."""
    return [
        (st.bay, st.row)
        for st in sorted(yard.stacks.values(), key=lambda s: (s.bay, s.row))
    ]


def priority_to_container_id(yard: Yard) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for st in yard.stacks.values():
        for cont in st.containers:
            mapping[int(cont.priority)] = int(cont.id)
    return mapping


def gurobi_on(value: Any) -> bool:
    """True if a Gurobi variable or constant is 1 in the incumbent."""
    if value is None:
        return False
    raw = value.X if hasattr(value, "X") else value
    try:
        return float(raw) > 0.5
    except (TypeError, ValueError):
        return False


def occupancy_from_yard(yard: Yard) -> Dict[int, Tuple[int, int]]:
    """Return {priority: (col, tier)} using the same stack order as IP models."""
    pos: Dict[int, Tuple[int, int]] = {}
    for col, st in enumerate(sorted(yard.stacks.values(), key=lambda s: (s.bay, s.row)), 1):
        for tier, cont in enumerate(st.containers, 1):
            pos[int(cont.priority)] = (col, tier)
    return pos


def sequential_retrieval_plan(yard: Yard) -> RelocationPlan:
    """Plan of only retrievals, in priority order (already-sorted yard)."""
    columns: Dict[Tuple[int, int], List[int]] = {
        (st.bay, st.row): [c.id for c in st.containers]
        for st in yard.stacks.values()
    }
    pri_to_id = priority_to_container_id(yard)
    plan = RelocationPlan()
    for pri in sorted(pri_to_id):
        cid = pri_to_id[pri]
        src = None
        for key, ids in columns.items():
            if cid in ids:
                src = key
                break
        if src is None:
            break
        plan.add(Movement(cid, src, None))
        columns[src] = [i for i in columns[src] if i != cid]
    return plan


def plan_to_moves_extra(plan: RelocationPlan) -> List[Dict[str, Any]]:
    return [
        {
            "container_id": m.container_id,
            "from": list(m.from_pos),
            "to": list(m.to_pos) if m.to_pos is not None else None,
            "kind": "retrieve" if m.to_pos is None else "relocate",
        }
        for m in plan.movements
    ]


def merge_solver_and_plan_metrics(
    solver: Mapping[str, Any],
    validated: Mapping[str, float],
) -> Dict[str, float]:
    """Keep official plan metrics (relocations, crane_time) plus solver stats."""
    out: Dict[str, float] = {str(k): float(v) for k, v in validated.items()}
    obj = solver.get("obj")
    out["solver_relocations"] = (
        float(obj) if obj is not None else float("inf")
    )
    out["optimal_proven"] = 1.0 if solver.get("optimal") else 0.0
    out["time_out"] = 1.0 if solver.get("time_out") else 0.0
    out["n_vars"] = float(solver.get("n_vars") or 0.0)
    out["n_constrs"] = float(solver.get("n_constrs") or 0.0)
    solve_t = solver.get("solve_time_s", solver.get("solve_time", 0.0))
    out["solve_time_s"] = float(solve_t or 0.0)
    if (
        float(validated.get("feasible", 0.0)) == 1.0
        and obj is not None
    ):
        out["count_match"] = float(
            float(validated.get("relocations", -1.0)) == float(obj)
        )
    else:
        out["count_match"] = 0.0
    return out


def plan_from_brp_ii_vars(
    yard: Yard,
    x_vals: Sequence[Tuple[int, int, int, int, int, int]],
    y_vals: Sequence[Tuple[int, int, int]],
    n: int,
) -> RelocationPlan:
    """Turn BRP-II x/y incumbents into an explicit relocate+retrieve plan."""
    keys = stack_keys(yard)
    pri_to_id = priority_to_container_id(yard)
    columns: List[List[int]] = [
        [c.id for c in yard.stacks[key].containers] for key in keys
    ]
    plan = RelocationPlan()
    y_by_t = {t: (i, j) for i, j, t in y_vals}

    for t in range(1, n):
        period = [(j, n_id, i, k) for i, j, k, _l, n_id, tt in x_vals if tt == t]
        period.sort(key=lambda item: -item[0])
        for _j, n_id, i, k in period:
            cid = pri_to_id[n_id]
            src = keys[i - 1]
            dst = keys[k - 1]
            plan.add(Movement(cid, src, dst))
            columns[i - 1].pop()
            columns[k - 1].append(cid)
        if t in y_by_t:
            i, _j = y_by_t[t]
            src = keys[i - 1]
        else:
            cid_t = pri_to_id[t]
            src = next(
                keys[idx] for idx, col in enumerate(columns) if cid_t in col
            )
        plan.add(Movement(pri_to_id[t], src, None))
        src_idx = keys.index(src)
        columns[src_idx].pop()

    if n >= 1 and n in pri_to_id:
        cid = pri_to_id[n]
        for idx, col in enumerate(columns):
            if cid in col:
                plan.add(Movement(cid, keys[idx], None))
                break
    return plan


def occupancy_stages_to_plan(
    occupancy: Mapping[int, Mapping[int, Tuple[int, int]]],
    y_vals: Mapping[Tuple[int, int], Any],
    n: int,
    keys: Sequence[Tuple[int, int]],
    pri_to_id: Mapping[int, int],
    max_tiers: Optional[int] = None,
) -> RelocationPlan:
    """
    Build a plan from per-stage occupancy {stage: {priority: (col, tier)}}
    and reshuffle indicators y[s, i].

    Some submodels leave the occupancy after the last modelled retrieval
    unconstrained; missing destinations are filled with a feasible stack
    that is not the current target column (restricted BRP).
    """
    plan = RelocationPlan()
    current = dict(occupancy.get(1) or {})
    n_cols = len(keys)

    def heights_from(pos: Mapping[int, Tuple[int, int]]) -> List[int]:
        h = [0] * n_cols
        for col, tier in pos.values():
            h[col - 1] = max(h[col - 1], int(tier))
        return h

    def dest_col(
        i: int,
        src_c: int,
        target_c: int,
        nxt: Mapping[int, Tuple[int, int]],
        heights: Sequence[int],
    ) -> int:
        if i in nxt:
            cand = int(nxt[i][0])
            if cand != src_c and cand != target_c:
                if max_tiers is None or heights[cand - 1] < max_tiers:
                    return cand
        for col in range(1, n_cols + 1):
            if col == src_c or col == target_c:
                continue
            if max_tiers is not None and heights[col - 1] >= max_tiers:
                continue
            return col
        for col in range(1, n_cols + 1):
            if col != src_c:
                return col
        return src_c

    for stage in range(1, n):
        if stage not in current:
            break
        nxt = dict(occupancy.get(stage + 1) or {})
        movers = [
            i for i in range(stage + 1, n + 1)
            if float(y_vals.get((stage, i), 0) or 0) > 0.5 and i in current
        ]
        if not movers and stage in current:
            target_c = int(current[stage][0])
            target_p = int(current[stage][1])
            movers = [
                i for i in range(stage + 1, n + 1)
                if i in current
                and int(current[i][0]) == target_c
                and int(current[i][1]) > target_p
            ]
        movers.sort(key=lambda i: -int(current[i][1]))
        target_c = int(current[stage][0])
        heights = heights_from(current)
        for i in movers:
            src_c, _src_p = current[i]
            dc = dest_col(i, int(src_c), target_c, nxt, heights)
            plan.add(Movement(
                int(pri_to_id[i]),
                keys[src_c - 1],
                keys[dc - 1],
            ))
            heights[src_c - 1] -= 1
            heights[dc - 1] += 1
            current[i] = (dc, heights[dc - 1])
        src_c, _src_p = current[stage]
        plan.add(Movement(int(pri_to_id[stage]), keys[src_c - 1], None))
        del current[stage]
        # Remaining containers keep columns; drop a tier if they sat above
        # the retrieved slot in the same stack (they were movers already).
    if n in current:
        src_c, _src_p = current[n]
        plan.add(Movement(int(pri_to_id[n]), keys[src_c - 1], None))
    elif n >= 1 and n in (occupancy.get(n) or {}):
        src_c, _src_p = occupancy[n][n]
        plan.add(Movement(int(pri_to_id[n]), keys[src_c - 1], None))
    return plan
