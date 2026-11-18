"""
------------------------------- Reference --------------------------------
C.-J. Ting, K.-C. Wu,
"Optimizing container relocation operations at container yards with
 beam search",
Transportation Research Part E 103 (2017) 17–31.
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
    return max(candidates, key=lambda index: (earliest(index), index))