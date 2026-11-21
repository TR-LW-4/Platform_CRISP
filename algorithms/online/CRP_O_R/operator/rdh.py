"""
Revised difference relocation rule.
<2010> <heuristic> <restricted> <single-bay> <CRP-O-R>

------------------------------- Reference --------------------------------
W. Zhao, A. V. Goodchild,
"The impact of truck arrival information on container terminal rehandling",
Transportation Research Part E 46 (2010) 327–343.
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


def _open(stacks, source, max_tiers):
    for index, stack in enumerate(stacks):
        if index != source and len(stack) < max_tiers:
            yield index, stack


def choose_destination(stacks, max_tiers, priority=lambda container: container):
    found = _locate(stacks, priority)
    if found is None:
        return None
    source, blocker = found

    selected = None
    smallest = INF
    for index, stack in _open(stacks, source, max_tiers):
        label = _earliest(stack, priority)
        if label > blocker and (label == INF or smallest > label):
            smallest = label
            selected = index
    if selected is not None:
        return selected

    smallest = INF
    for index, stack in _open(stacks, source, max_tiers):
        label = _earliest(stack, priority)
        if label == blocker and smallest > label:
            smallest = label
            selected = index
    if selected is not None:
        return selected

    largest = -INF
    for index, stack in _open(stacks, source, max_tiers):
        top = -INF + 1 if not stack else priority(stack[-1])
        if top < blocker and largest < top:
            largest = top
            selected = index
    if selected is not None:
        return selected

    smallest_top = INF
    for index, stack in _open(stacks, source, max_tiers):
        top = INF - 1 if not stack else priority(stack[-1])
        if smallest_top > top:
            smallest_top = top
            selected = index
    return selected
