"""
Expected reshuffle-index relocation rule.
<2016> <heuristic> <restricted> <single-bay> <CRP-O-R>

------------------------------- Reference --------------------------------
D. Ku, T. S. Arthanari,
"Container relocation problem with time windows for container departure",
European Journal of Operational Research 252 (2016) 1031–1039.
A container with the same priority as the blocker contributes 0.5.
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


def choose_destination(stacks, max_tiers, priority=lambda container: container):
    found = _locate(stacks, priority)
    if found is None:
        return None
    source, blocker = found
    best = INF
    chosen = None
    for index, stack in enumerate(stacks):
        if index == source or len(stack) >= max_tiers:
            continue
        score = 0.0
        for container in stack:
            rank = priority(container)
            if rank < blocker:
                score += 1
            elif rank == blocker:
                score += 0.5
        if score < best:
            best = score
            chosen = index
    return chosen
