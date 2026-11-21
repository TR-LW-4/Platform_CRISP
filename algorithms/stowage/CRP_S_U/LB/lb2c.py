"""
LB2c for the block relocation problem with a stowage plan.
Counts 2-blocking containers. Valid for the unrestricted and restricted variants.

------------------------------- Reference --------------------------------
S. Tanaka, S. Voß,
"An exact algorithm for the block relocation problem with a stowage plan",
European Journal of Operational Research 279 (2019) 767–781. Section 5.1.1.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations


def two_cycle(upper, lower):
    return upper[0] == lower[0] and lower[1] < upper[1]


def blocking_positions(stacks):
    """Map each container to ``(stack, index)`` and collect 2-blocking ones."""
    positions = {}
    blocking2 = set()
    for stack_index, stack in enumerate(stacks):
        for index, container in enumerate(stack):
            positions[container] = (stack_index, index)
            for below in stack[:index]:
                if two_cycle(container, below):
                    blocking2.add(container)
                    break
    return positions, blocking2


def lower_bound(stacks, heights, cache=None):
    """Number of 2-blocking containers still in the yard."""
    del heights, cache
    _, blocking2 = blocking_positions(stacks)
    return len(blocking2)
