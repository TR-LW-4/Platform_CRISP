"""
LB3 lower bound for CRP-U

------------------------------- Reference --------------------------------
F. Tricoire, J. Scagnetti, A. Beham,
"New insights on the block relocation problem",
Computers & Operations Research 89 (2018) 127–139.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from algorithms.distinct.CRP_U.LB.lb1_u import lb1_u


def lb3_u(stacks, max_tiers) -> int:
    lower_bound = lb1_u(stacks, max_tiers)
    priorities = [priority for stack in stacks for priority in stack]
    if not priorities:
        return lower_bound
    target = min(priorities)
    lows = [min(stack) if stack else max(priorities) + 1 for stack in stacks]
    shortest = min(len(stack) for stack in stacks)
    level = 0
    while level < shortest:
        max_low = max(min(stack[:len(stack) - level]) for stack in stacks)
        min_top = max(priorities) + 1
        for stack, low in zip(stacks, lows):
            top = stack[len(stack) - 1 - level]
            if top == target:
                min_top = target
                break
            if low < top < min_top:
                min_top = top
        if min_top <= max_low:
            break
        level += 1
    return lower_bound + level
