"""
LB2 lower bound for CRP-U

------------------------------- Reference --------------------------------
F. Forster, A. Bortfeldt,
"A tree search procedure for the container relocation problem",
Computers & Operations Research 39 (2012) 299–309.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from algorithms.distinct.CRP_U.LB.lb1_u import lb1_u


def lb2_u(stacks, max_tiers) -> int:
    state = _retrieve(stacks)
    lower_bound = lb1_u(state, max_tiers)
    if not all(state):
        return lower_bound
    min_top = max(priority for stack in state for priority in stack) + 1
    max_low = 0
    for stack in state:
        low = min(stack)
        if low < stack[-1] < min_top:
            min_top = stack[-1]
        max_low = max(max_low, low)
    return lower_bound + 1 if min_top > max_low else lower_bound


def _retrieve(stacks):
    state = [list(stack) for stack in stacks]
    while any(state):
        target = min(min(stack) for stack in state if stack)
        source = next((stack for stack in state if stack and stack[-1] == target), None)
        if source is None:
            break
        source.pop()
    return state
