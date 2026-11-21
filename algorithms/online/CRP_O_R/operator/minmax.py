"""
MinMax relocation rule.
<2012> <heuristic> <restricted> <single-bay> <CRP-O-R>

------------------------------- Reference --------------------------------
W. Zhu, H. Qin, A. Lim, H. Zhang,
"Iterative deepening A* algorithms for the container relocation problem",
IEEE Transactions on Automation Science and Engineering 9 (2012) 710–722.
The C++ implementation used here is the PR2-style MinMax: an empty stack
replaces any safe stack, and the last empty stack is kept.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

INF = 10 ** 18


def _locate(stacks, priority):
    best = None
    source = None
    for index, stack in enumerate(stacks):
        for container in stack:
            rank = priority(container)
            if best is None or rank < best:
                best = rank
                source = index
    if source is None or priority(stacks[source][-1]) == best:
        return None
    return source, priority(stacks[source][-1])


def _earliest(stack, priority):
    if not stack:
        return INF
    return min(priority(container) for container in stack)


def _safe(stacks, source, blocker, max_tiers, priority):
    lowest_index = INF
    lowest = None
    for index, stack in enumerate(stacks):
        if index == source or len(stack) >= max_tiers:
            continue
        selected = _earliest(stack, priority)
        if selected > blocker:
            if lowest_index > selected:
                lowest_index = selected
                lowest = index
            elif selected == INF:
                lowest_index = 0
                lowest = index
    return lowest


def _unsafe(stacks, source, blocker, max_tiers, priority):
    largest_index = -INF
    largest = None
    for index, stack in enumerate(stacks):
        if index == source or len(stack) >= max_tiers:
            continue
        selected = _earliest(stack, priority)
        if selected <= blocker and largest_index < selected:
            largest_index = selected
            largest = index
    return largest


def choose_destination(stacks, max_tiers, priority=lambda container: container):
    found = _locate(stacks, priority)
    if found is None:
        return None
    source, blocker = found
    safe = _safe(stacks, source, blocker, max_tiers, priority)
    if safe is not None:
        return safe
    return _unsafe(stacks, source, blocker, max_tiers, priority)
