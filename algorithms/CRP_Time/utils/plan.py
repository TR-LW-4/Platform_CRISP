"""
From a sequence of moves on the stack lists to a platform plan, and from a plan
to the problem's actions.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from core.plan import Movement, RelocationPlan


def replay(stacks: Sequence[Sequence[int]], positions: Sequence[Tuple[int, int]],
           moves: Sequence[Tuple[int, Optional[int]]]) -> RelocationPlan:
    """Plan of the moves (src, dst) on stack indices, dst None for a retrieval.

    Each move takes the top container of src. Tiers are 1-based: the source tier
    before the move and the destination tier after it. Only an empty source is
    refused here; the problem's validate_plan checks the rules of the problem.

    Example:
        plan = replay(stacks, positions, [(0, 2), (0, None)])
    """
    st = [list(s) for s in stacks]
    plan = RelocationPlan()
    for src, dst in moves:
        if not st[src]:
            raise ValueError(f"move ({src}, {dst}): stack {src} is empty")
        c = st[src].pop()
        if dst is None:
            plan.add(Movement(c, positions[src], None, from_tier=len(st[src]) + 1))
        else:
            st[dst].append(c)
            plan.add(Movement(c, positions[src], positions[dst],
                              from_tier=len(st[src]) + 1, to_tier=len(st[dst])))
    return plan


def actions(plan: RelocationPlan, num_rows: int) -> List[int]:
    """The problem's actions for a plan: the flat index of each relocation's destination."""
    return [(m.to_pos[0] - 1) * num_rows + (m.to_pos[1] - 1) for m in plan.movements if not m.is_retrieval]
