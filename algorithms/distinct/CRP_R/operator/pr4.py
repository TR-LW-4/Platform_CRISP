"""
------------------------------- Reference --------------------------------
W. Zhu, H. Qin, A. Lim, H. Zhang,
"Iterative deepening A* algorithms for the container relocation problem",
IEEE Transactions on Automation Science and Engineering 9 (2012) 710–722.
Zhu et al. (2012) call this probe heuristic PR4.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from typing import Optional


def choose_destination(stacks, max_tiers) -> Optional[int]:
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

    good = [index for index in candidates if earliest(index) > blocker]
    if good:
        return min(good, key=lambda index: (earliest(index), index))

    ranked = sorted(candidates, key=lambda index: (-earliest(index), index))
    best = ranked[0]
    second = blocker == sorted(stacks[source])[1]
    one_slot = max_tiers - len(stacks[best]) == 1
    if not second and one_slot and len(ranked) > 1:
        return ranked[1]
    return best
