"""
PU1 unrestricted relocation rule.
<2012> <heuristic> <unrestricted> <single-bay> <CRP-O-U>

------------------------------- Reference --------------------------------
W. Zhu, H. Qin, A. Lim, H. Zhang,
"Iterative deepening A* algorithms for the container relocation problem",
IEEE Transactions on Automation Science and Engineering 9 (2012) 710–722.
A preparatory move is made only from a stack of height one. The C++
condition assigns ``secondSmallest`` instead of comparing it, so taller
stacks never enter that branch.
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


def _safe(stacks, source, blocker, max_tiers, priority, skip=()):
    lowest_index = INF
    lowest = None
    for index, stack in enumerate(stacks):
        if index == source or index in skip or len(stack) >= max_tiers:
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


def _prepare(stacks, source, max_tiers, priority):
    """Move one height-one container aside, then the blocker into its place."""
    selected = None
    best = -1
    for index, stack in enumerate(stacks):
        if index == source or len(stack) != 1:
            continue
        rank = priority(stack[0])
        if rank > best:
            best = rank
            selected = index
    if selected is None:
        return None
    destination = _safe(stacks, source, best, max_tiers, priority, skip={selected})
    if destination is None:
        return None
    return [(selected, destination), (source, selected)]


def choose_moves(stacks, max_tiers, priority=lambda container: container):
    found = _locate(stacks, priority)
    if found is None:
        return []
    source, blocker = found
    safe = _safe(stacks, source, blocker, max_tiers, priority)
    if safe is not None:
        return [(source, safe)]
    prepared = _prepare(stacks, source, max_tiers, priority)
    if prepared is not None:
        return prepared
    unsafe = _unsafe(stacks, source, blocker, max_tiers, priority)
    if unsafe is None:
        return []
    return [(source, unsafe)]
