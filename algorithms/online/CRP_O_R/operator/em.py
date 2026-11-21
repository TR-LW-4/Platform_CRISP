"""
Expected MinMax relocation rule.
<2018> <heuristic> <restricted> <single-bay> <CRP-O-R>

------------------------------- Reference --------------------------------
V. Galle,
"Optimization models and methods for storage yard operations in maritime
container terminals", Ph.D. thesis, 2018.
Tied stacks are broken by the lowest height. When the best minimum label
equals the blocker, the stack containing fewer copies of that label is kept.
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


def _shallowest(stacks, indexes):
    if not indexes:
        return None
    return min(indexes, key=lambda index: (len(stacks[index]), index))


def choose_destination(stacks, max_tiers, priority=lambda container: container):
    found = _locate(stacks, priority)
    if found is None:
        return None
    source, blocker = found
    lowest = INF
    safe = []
    for index, stack in enumerate(stacks):
        if index == source or len(stack) >= max_tiers:
            continue
        label = _earliest(stack, priority)
        if label > blocker:
            if label < lowest:
                lowest = label
                safe = [index]
            elif label == lowest:
                safe.append(index)
    if safe:
        return _shallowest(stacks, safe)

    best = -INF
    fallback = []
    for index, stack in enumerate(stacks):
        if index == source or len(stack) >= max_tiers:
            continue
        label = _earliest(stack, priority)
        if label > best:
            best = label
            fallback = [index]
        elif label == best:
            fallback.append(index)
    if not fallback:
        return None
    if best != blocker:
        return _shallowest(stacks, fallback)

    fewest = INF
    chosen = []
    for index in fallback:
        copies = sum(1 for container in stacks[index] if priority(container) == best)
        if copies < fewest:
            fewest = copies
            chosen = [index]
        elif copies == fewest:
            chosen.append(index)
    return _shallowest(stacks, chosen)
