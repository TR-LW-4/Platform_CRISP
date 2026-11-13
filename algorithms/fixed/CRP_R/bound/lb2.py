"""
------------------------------- Reference --------------------------------
W. Zhu, H. Qin, A. Lim, H. Zhang,
"Iterative deepening A* algorithms for the container relocation problem",
IEEE Transactions on Automation Science and Engineering 9 (2012) 710–722.
LB2 is LB1 plus one extra move for each blocker of the current target
that has no safe stack.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from typing import Sequence

from algorithms.CRP_R.bound.lb1 import lb1


def lb2(stacks: Sequence[Sequence[int]], max_tiers: int) -> int:
    total = lb1(stacks)
    present = [priority for stack in stacks for priority in stack]
    if not present:
        return total
    target = min(present)
    source = next(index for index, stack in enumerate(stacks) if target in stack)
    tier = stacks[source].index(target)
    for priority in stacks[source][tier + 1:]:
        if not _has_safe_stack(stacks, max_tiers, source, priority):
            total += 1
    return total


def _has_safe_stack(stacks: Sequence[Sequence[int]], max_tiers: int, source: int, priority: int) -> bool:
    for index, stack in enumerate(stacks):
        if index == source or len(stack) >= max_tiers:
            continue
        if not stack or min(stack) > priority:
            return True
    return False
