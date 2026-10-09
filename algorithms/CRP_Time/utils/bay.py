"""
The bay of a CRP-Time instance as lists of stacks.

A stack is a list of retrieval priorities from bottom to top. Stacks are indexed
by the flat stack index 0 .. S-1 of the problem's actions, and positions[i] is
the (bay, row) of stack i.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List, Sequence, Tuple


def read_bay(env) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    """Reset env and return (stacks, positions) of the initial bay.

    Containers that are on top at the start stay in the bay: their retrievals
    belong to the plan and to its crane time, so the problem's automatic
    retrieval is skipped.

    Example:
        stacks, positions = read_bay(problem_factory())
    """
    env.reset(options={"skip_auto_retrieve": True})
    positions = [env._action_to_stack(a) for a in range(env.config.num_bays * env.config.num_rows)]
    stacks = [
        [int(c.priority) for c in env.yard.stacks[p].containers] if p in env.yard.stacks else []
        for p in positions
    ]
    return stacks, positions


def target_stack(stacks: Sequence[Sequence[int]], c: int) -> int:
    """Index of the stack that holds container c, -1 if none does."""
    return next((i for i, st in enumerate(stacks) if c in st), -1)


def free_destinations(stacks: Sequence[Sequence[int]], src: int, max_tiers: int) -> List[int]:
    """The stacks other than src with room for one more container, in index order."""
    return [i for i, st in enumerate(stacks) if i != src and len(st) < max_tiers]
