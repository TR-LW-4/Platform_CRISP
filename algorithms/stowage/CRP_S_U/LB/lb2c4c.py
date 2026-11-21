"""
LB2c4c for the block relocation problem with a stowage plan.
LB2c plus a set-cover bound on minimal 4-cycles. Section 5.1.2.

------------------------------- Reference --------------------------------
S. Tanaka, S. Voß,
"An exact algorithm for the block relocation problem with a stowage plan",
European Journal of Operational Research 279 (2019) 767–781.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from algorithms.stowage.CRP_S_U.LB.lb2c import blocking_positions, two_cycle


def cycle_sets(stacks):
    """Return ``(B2, B4, pairs)`` for the yard stacks.

    ``B4`` contains 4-blocking containers of minimal 4-cycles whose
    4-blocking containers are not 2-blocking. ``pairs`` are those
    ``{blocking, blocking}`` pairs, one per distinct pair rather than one
    per cycle. Covering the pairs still covers every such cycle.
    """
    positions, blocking2 = blocking_positions(stacks)
    pairs = set()
    containers = list(positions)
    for stack in stacks:
        for index, container in enumerate(stack):
            if container in blocking2:
                continue
            for below in stack[:index]:
                for other in containers:
                    if other in blocking2 or other == container or other == below:
                        continue
                    if other[0] != below[0] or other[1] <= below[1]:
                        continue
                    other_stack, other_index = positions[other]
                    for fourth in stacks[other_stack][:other_index]:
                        if fourth in (container, below, other):
                            continue
                        if fourth[0] != container[0] or fourth[1] >= container[1]:
                            continue
                        nodes = (container, below, other, fourth)
                        if any(
                            _stacked_two_cycle(nodes[a], nodes[b], positions)
                            for a in range(4) for b in range(a + 1, 4)
                        ):
                            continue
                        pair = frozenset((container, other))
                        if len(pair) == 2:
                            pairs.add(pair)

    blocking4 = set()
    for pair in pairs:
        blocking4.update(pair)
    return blocking2, blocking4, pairs


def lower_bound(stacks, heights, cache=None):
    """LB2c plus the set-cover bound on minimal 4-cycles."""
    del heights, cache
    blocking2, _, pairs = cycle_sets(stacks)
    return len(blocking2) + _cover_count(pairs)


def _stacked_two_cycle(left, right, positions):
    stack_left, index_left = positions[left]
    stack_right, index_right = positions[right]
    if stack_left != stack_right:
        return False
    if index_left > index_right:
        return two_cycle(left, right)
    return two_cycle(right, left)


def _cover_count(pairs):
    """Minimum-set-cover lower bound on the 4-blocking containers."""
    if not pairs:
        return 0
    degree = {}
    for pair in pairs:
        for container in pair:
            degree[container] = degree.get(container, 0) + 1
    ordered = sorted(degree.values(), reverse=True)
    covered = 0
    count = 0
    for value in ordered:
        covered += value
        count += 1
        if covered >= len(pairs):
            return count
    return count
