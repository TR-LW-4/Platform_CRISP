"""Yard layout for the stowage-plan relocation problem.

The layout is the yard: each stack is a list of containers from bottom to top.
A container is the pair ``(vessel stack, vessel tier)``. Tiers in a vessel
stack are filled from 0 upward, so the next tier of a stack is the smallest
tier that is still in the yard. The vessel is not stored separately; containers
already retrieved are the ones already loaded.
"""

from __future__ import annotations


def freeze(stacks):
    return tuple(tuple(stack) for stack in stacks)


def loaded_from(stacks, heights):
    """Next open tier of every vessel stack."""
    nxt = list(heights)
    for stack in stacks:
        for vessel_stack, vessel_tier in stack:
            if vessel_tier < nxt[vessel_stack]:
                nxt[vessel_stack] = vessel_tier
    return nxt


def normalize(stacks, heights):
    """Retrieve every container that is already on top of its vessel slot.

    The number of yard stacks is preserved, including stacks that become empty.
    """
    state = [list(stack) for stack in stacks]
    progressed = True
    while progressed:
        progressed = False
        loaded = loaded_from(state, heights)
        for stack in state:
            if not stack:
                continue
            vessel_stack, vessel_tier = stack[-1]
            if vessel_tier != loaded[vessel_stack]:
                continue
            stack.pop()
            progressed = True
            break
    return state


def top_target(stack, loaded):
    """Index of the highest container whose vessel slot is already open."""
    for index in range(len(stack) - 1, -1, -1):
        vessel_stack, vessel_tier = stack[index]
        if vessel_tier == loaded[vessel_stack]:
            return index
    return None
