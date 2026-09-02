"""
Relocation replay helpers for TanakaVossIP2022.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List, Tuple

from core.plan import Movement, RelocationPlan
from core.yard import Yard
from algorithms.CRP_R.solver_ip.common import (
    priority_to_container_id,
    stack_keys,
)


def _plan_from_tanaka_relocations(
    yard: Yard,
    records: List[Tuple[int, int, int]],
) -> RelocationPlan:
    """Replay Tanaka ``src->dst`` relocations plus sequential retrievals."""
    keys = stack_keys(yard)
    pri_to_id = priority_to_container_id(yard)
    columns: List[List[int]] = [
        [c.id for c in yard.stacks[key].containers] for key in keys
    ]
    plan = RelocationPlan()
    ri = 0
    n = len(pri_to_id)
    for t in range(1, n + 1):
        cid = pri_to_id[t]
        while True:
            src_idx = next(
                idx for idx, col in enumerate(columns) if cid in col
            )
            if columns[src_idx][-1] == cid:
                plan.add(Movement(cid, keys[src_idx], None))
                columns[src_idx].pop()
                break
            if ri >= len(records):
                break
            pri, src, dst = records[ri]
            ri += 1
            mover = pri_to_id[pri]
            plan.add(Movement(mover, keys[src - 1], keys[dst - 1]))
            columns[src - 1].pop()
            columns[dst - 1].append(mover)
    return plan
