"""
Expected grouping relocation rule.
<2018> <heuristic> <restricted> <single-bay> <CRP-OG-R>

------------------------------- Reference --------------------------------
V. Galle,
"Optimization models and methods for storage yard operations in maritime
container terminals", Ph.D. thesis, 2018.
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


def choose_destination(stacks, max_tiers, priority=lambda container: container):
    found = _locate(stacks, priority)
    if found is None:
        return None
    source, blocker = found
    best_count = INF
    best_label = -INF
    chosen = None
    for index, stack in enumerate(stacks):
        if index == source or len(stack) >= max_tiers:
            continue
        count = sum(1 for container in stack if priority(container) < blocker)
        label = _earliest(stack, priority)
        if count < best_count or (count == best_count and label > best_label):
            best_count = count
            best_label = label
            chosen = index
    return chosen
