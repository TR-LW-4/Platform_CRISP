"""
------------------------------- Reference --------------------------------
K.H. Kim, G.P. Hong,
"A heuristic rule for relocating blocks",
Computers & Operations Research 33 (2006) 940–954.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from functools import cache
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

    max_priority = max(priorities)
    remaining = len(priorities)
    stack_count = len(stacks)
    rounded_mean = (remaining + stack_count - 1) // stack_count

    @cache
    def expected_relocations(k, n, T):
        if k == 0:
            return 0.0

        outside = max_priority - T + k
        later = outside - (n - 1)
        total = sum(expected_relocations(k - 1, priority, T) for priority in range(1, n))
        if later > 0:
            total += later * (1 + expected_relocations(k - 1, n, T))
        return total / outside

    def stack_enar(height, earliest):
        if height * stack_count > remaining:
            expected_height = height + 1
        else:
            expected_height = rounded_mean

        T = min(expected_height, max_tiers, max_priority)
        return expected_relocations(max(0, T - height), earliest, T)

    def score(index):
        stack = stacks[index]
        height = len(stack)
        earliest = min(stack, default=max_priority + 1)

        before = stack_enar(height, earliest)
        after = stack_enar(height + 1, min(earliest, blocker))
        confirmed = int(blocker > earliest)
        return after - before + confirmed, index

    return min(candidates, key=score, default=None)