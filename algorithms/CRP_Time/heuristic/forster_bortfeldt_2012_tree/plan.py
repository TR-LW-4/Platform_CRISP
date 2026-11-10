"""
Independent check of a move sequence for the setting of Forster & Bortfeldt
(2012), p. 302: only top items move, a remove takes an item of the currently
lowest group, no stack exceeds H, and the bay is empty at the end. Uses plain
lists, not state.Bay.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite
F. Forster, A. Bortfeldt, "A tree search procedure for the container
relocation problem", Computers & Operations Research 39 (2012) 299-309.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple


def check_plan(stacks: Sequence[Sequence[int]], H: int, plan: Sequence[Tuple[int, Optional[int], int]]) -> int:
    """Replays plan, moves (d, r, g) with r None for a remove; raises ValueError on the first violation.
    Returns the number of relocations."""
    bay = [list(s) for s in stacks]
    if any(len(s) > H for s in bay):
        raise ValueError("initial layout higher than H")
    if not plan and any(bay):
        raise ValueError("empty plan for a non-empty layout")
    relocations = 0
    for i, (d, r, g) in enumerate(plan):
        if not 0 <= d < len(bay) or not bay[d]:
            raise ValueError(f"move {i}: source stack {d} empty or unknown")
        if bay[d][-1] != g:
            raise ValueError(f"move {i}: item {g} is not on top of stack {d}")
        if r is None:
            if g != min(min(s) for s in bay if s):
                raise ValueError(f"move {i}: remove of {g} out of group order")
            bay[d].pop()
            continue
        if not 0 <= r < len(bay) or r == d:
            raise ValueError(f"move {i}: bad target stack {r}")
        if len(bay[r]) >= H:
            raise ValueError(f"move {i}: stack {r} is full (H = {H})")
        bay[r].append(bay[d].pop())
        relocations += 1
    if any(bay):
        raise ValueError("bay not empty at the end")
    return relocations
