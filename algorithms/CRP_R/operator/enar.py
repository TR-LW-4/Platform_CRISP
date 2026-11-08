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

from collections.abc import Sequence
from functools import cache

def choose_destination(stacks, max_tiers):
    priorities = [p for stack in stacks for p in stack]
    if not priorities:
        return None

    target = min(priorities)
    src = next(i for i, stack in enumerate(stacks) if target in stack)
    blocker = stacks[src][-1]
    if blocker == target:
        return None

    candidates = [
        i for i, stack in enumerate(stacks)
        if i != src and len(stack) < max_tiers
    ]

    initial_count = max(priorities)
    remaining = len(priorities)
    stack_count = len(stacks)
    rounded_mean = (remaining + stack_count - 1) // stack_count

    @cache
    def expected_relocations(k, n, T):
        if k == 0:
            return 0.0

        outside = initial_count - T + k
        later = outside - (n - 1)
        total = sum(expected_relocations(k - 1, p, T) for p in range(1, n))
        if later > 0:
            total += later * (1 + expected_relocations(k - 1, n, T))
        return total / outside

    def stack_enar(height, earliest):
        if height * stack_count > remaining:
            expected_height = height + 1
        else:
            expected_height = rounded_mean

        T = min(expected_height, max_tiers, initial_count)
        return expected_relocations(max(0, T - height), earliest, T)

    def score(dst):
        stack = stacks[dst]
        height = len(stack)
        earliest = min(stack, default=initial_count + 1)

        before = stack_enar(height, earliest)
        after = stack_enar(height + 1, min(earliest, blocker))
        confirmed = int(blocker > earliest)
        return after - before + confirmed, dst

    return min(candidates, key=score, default=None)