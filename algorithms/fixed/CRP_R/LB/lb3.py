"""
------------------------------- Reference --------------------------------
W. Zhu, H. Qin, A. Lim, H. Zhang,
"Iterative deepening A* algorithms for the container relocation problem",
IEEE Transactions on Automation Science and Engineering 9 (2012) 710–722.
LB3 is LB1 plus unavoidable extra moves for every stack's next container.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from algorithms.fixed.CRP_R.LB.lb1 import lb1
from algorithms.fixed.CRP_R.LB.lb2 import _has_safe_stack


def lb3(stacks, max_tiers) -> int:
    lower_bound = lb1(stacks, max_tiers)
    for source, stack in enumerate(stacks):
        if not stack:
            continue
        target = min(stack)
        tier = stack.index(target)
        blockers = stack[tier + 1:]
        if not blockers:
            continue
        reduced = []
        for current in stacks:
            cut = len(current)
            for slot, priority in enumerate(current):
                if priority < target:
                    cut = slot
                    break
            reduced.append(current[:cut])
        for priority in blockers:
            if not _has_safe_stack(reduced, max_tiers, source, priority):
                lower_bound += 1
    return lower_bound