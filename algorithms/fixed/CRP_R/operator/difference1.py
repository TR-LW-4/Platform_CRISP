"""
------------------------------- Reference --------------------------------
T. Ünlüyurt, C. Aydın,
"Improved rehandling strategies for the container retrieval process",
Journal of Advanced Transportation 46 (2012) 378–393.
The destination rule is Difference1 as used by Bacci, Mattia and Ventura,
Computers & Operations Research 103 (2019) 252–264.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Optional


def choose_destination(stacks, max_tiers) -> Optional[int]:
    """Destination of the current top blocker under Difference1."""
    present = [item for stack in stacks for item in stack]
    if not present:
        return None
    n = max(present)
    target = min(present)
    source = next(index for index, stack in enumerate(stacks) if target in stack)
    if stacks[source][-1] == target:
        return None

    blocker = stacks[source][-1]
    best_top = -1
    best_good = smallest_top = n * n
    bad = good = smallest = -1
    for index, stack in enumerate(stacks):
        if index == source or len(stack) >= max_tiers:
            continue
        earliest = min(stack) if stack else n + 1
        if earliest < blocker:
            top = stack[-1]
            if top < blocker and top > best_top:
                best_top = top
                bad = index
            if top < smallest_top:
                smallest_top = top
                smallest = index
        elif earliest < best_good:
            best_good = earliest
            good = index
    if good != -1:
        return good
    if bad != -1:
        return bad
    if smallest != -1:
        return smallest
    return None
