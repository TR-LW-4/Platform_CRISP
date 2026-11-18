"""
------------------------------- Reference --------------------------------
M. Caserta, S. Schwarze, S. Voß,
"A New Binary Description of the Blocks Relocation Problem and Benefits
 in a Look Ahead Heuristic",
Evolutionary Computation in Combinatorial Optimization (EvoCOP 2009),
LNCS 5482, pp. 37–48, Springer, 2009.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from typing import Optional


def _min_max(stacks, max_tiers) -> Optional[int]:
    priorities = [priority for stack in stacks for priority in stack]
    if not priorities:
        return None

    target = min(priorities)
    source = next(index for index, stack in enumerate(stacks) if target in stack)
    blocker = stacks[source][-1]
    if blocker == target:
        return None

    candidates = [
        index for index, stack in enumerate(stacks)
        if index != source and len(stack) < max_tiers
    ]
    if not candidates:
        return None

    max_priority = max(priorities)

    def earliest(index):
        stack = stacks[index]
        return min(stack) if stack else max_priority + 1

    good = [index for index in candidates if earliest(index) > blocker]
    if good:
        return min(good, key=lambda index: (earliest(index), index))
    return min(candidates, key=lambda index: (-earliest(index), index))


def _completion(stacks, max_tiers):
    state = [list(stack) for stack in stacks]
    count = 0
    while True:
        priorities = [priority for stack in state for priority in stack]
        if not priorities:
            return count
        target = min(priorities)
        source = next(index for index, stack in enumerate(state) if target in stack)
        if state[source][-1] == target:
            state[source].pop()
            continue
        destination = _min_max(state, max_tiers)
        if destination is None:
            return count
        state[destination].append(state[source].pop())
        count += 1


def choose_destination(stacks, max_tiers) -> Optional[int]:
    priorities = [priority for stack in stacks for priority in stack]
    if not priorities:
        return None

    target = min(priorities)
    source = next(index for index, stack in enumerate(stacks) if target in stack)
    if stacks[source][-1] == target:
        return None

    candidates = [
        index for index, stack in enumerate(stacks)
        if index != source and len(stack) < max_tiers
    ]
    if not candidates:
        return None

    def score(index):
        trial = [list(stack) for stack in stacks]
        trial[index].append(trial[source].pop())
        return _completion(trial, max_tiers), index

    return min(candidates, key=score)