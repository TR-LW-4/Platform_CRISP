"""
Solution condensation for GLAH

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""


def condense(stacks, steps):
    """Merge a relocation of a container into its previous relocation when
    the new destination stack is untouched in between."""
    state, label = [], 0
    for stack in stacks:
        state.append(list(range(label, label + len(stack))))
        label += len(stack)
    condensed, last = [], {}
    for source, destination in steps:
        item = state[source].pop()
        if destination is None:
            condensed.append((source, None))
            continue
        state[destination].append(item)
        previous = last.get(item)
        if previous is not None and all(destination not in step for step in condensed[previous + 1:]):
            condensed[previous] = (condensed[previous][0], destination)
        else:
            last[item] = len(condensed)
            condensed.append((source, destination))
    return [(source, destination) for source, destination in condensed if source != destination]
