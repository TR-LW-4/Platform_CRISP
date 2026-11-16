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

from algorithms.fixed.CRP_R.LB.lb1 import lb1


def lb2(stacks, max_tiers) -> int:
    lower_bound = lb1(stacks, max_tiers)
    priorities = [priority for stack in stacks for priority in stack]
    if not priorities:
        return lower_bound
    target = min(priorities)
    source = next(index for index, stack in enumerate(stacks) if target in stack)
    tier = stacks[source].index(target)
    for priority in stacks[source][tier + 1:]:
        if not _has_safe_stack(stacks, max_tiers, source, priority):
            lower_bound += 1
    return lower_bound


def _has_safe_stack(stacks, max_tiers, source, priority) -> bool:
    for index, stack in enumerate(stacks):
        if index == source or len(stack) >= max_tiers:
            continue
        if not stack or min(stack) > priority:
            return True
    return False