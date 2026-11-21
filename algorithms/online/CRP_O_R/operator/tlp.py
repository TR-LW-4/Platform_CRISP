"""
Lowest-position relocation rule.
<2010> <heuristic> <restricted> <single-bay> <CRP-O-R>

------------------------------- Reference --------------------------------
Y. Lee, Y. J. Lee,
"A heuristic for retrieving containers from a yard",
Computers and Operations Research 37 (2010) 1139–1147.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

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


def choose_destination(stacks, max_tiers, priority=lambda container: container):
    found = _locate(stacks, priority)
    if found is None:
        return None
    source, _blocker = found
    chosen = None
    shortest = max_tiers + 1
    for index, stack in enumerate(stacks):
        if index == source or len(stack) >= max_tiers:
            continue
        if len(stack) < shortest:
            shortest = len(stack)
            chosen = index
    return chosen
