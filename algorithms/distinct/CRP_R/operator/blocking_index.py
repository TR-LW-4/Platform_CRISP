"""
------------------------------- Reference --------------------------------
L. Tang, W. Jiang, J. Liu, Y. Dong,
"Research into container reshuffling and stacking problems in container
 terminal yards",
IIE Transactions 47 (2015) 751–766.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from typing import Optional


def choose_destination(stacks, max_tiers, rule="H2") -> Optional[int]:
    priorities = [priority for stack in stacks for priority in stack]
    if not priorities:
        return None

    target = min(priorities)
    source = next(index for index, stack in enumerate(stacks) if target in stack)
    blocker = stacks[source][-1]
    if blocker == target:
        return None

    candidates = [
        index for index, stack in enumerate(stacks)
        if index != source and len(stack) < max_tiers
    ]
    if not candidates:
        return None

    max_priority = max(priorities)

    def earliest(index):
        stack = stacks[index]
        return min(stack) if stack else max_priority + 1

    def reshuffle_index(index):
        return sum(priority < blocker for priority in stacks[index])

    def blocking_index(index):
        placed = [*stacks[index], blocker]
        urgent_at = placed.index(min(placed))
        return len(placed) - 1 - urgent_at

    good = [index for index in candidates if earliest(index) > blocker]
    if good:
        return min(good, key=lambda index: (earliest(index), index))

    if rule == "H1":
        return min(candidates, key=lambda index: (reshuffle_index(index), -earliest(index), index))
    return min(candidates, key=lambda index: (blocking_index(index), -earliest(index), index))