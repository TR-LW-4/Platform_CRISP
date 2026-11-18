"""
SM-1 relocation rule for CRP-U

------------------------------- Reference --------------------------------
F. Tricoire, J. Scagnetti, A. Beham,
"New insights on the block relocation problem",
Computers & Operations Research 89 (2018) 127–139.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from typing import Optional, Tuple


def choose_move(stacks, max_tiers) -> Optional[Tuple[int, int]]:
    max_priority = max((max(stack) for stack in stacks if stack), default=0)
    if max_priority == 0:
        return None
    earliest = [min(stack) if stack else max_priority + 1 for stack in stacks]
    target = min(earliest)
    source = earliest.index(target)
    if stacks[source][-1] == target:
        return None
    move = _best_safe(stacks, max_tiers, earliest, max_priority)
    if move is not None:
        return move
    destination = _la_destination(stacks, max_tiers, earliest, source, stacks[source][-1])
    return None if destination is None else (source, destination)


def _best_safe(stacks, max_tiers, earliest, max_priority):
    best, best_gap = None, max_priority + 1
    for source, stack in enumerate(stacks):
        if len(stack) < 2 or stack[-1] == earliest[source]:
            continue
        blocker = stack[-1]
        for destination in range(len(stacks)):
            if (destination != source and len(stacks[destination]) < max_tiers
                    and earliest[destination] >= blocker
                    and earliest[destination] - blocker < best_gap):
                best, best_gap = (source, destination), earliest[destination] - blocker
    return best


def _la_destination(stacks, max_tiers, earliest, source, blocker):
    best = None
    for index, stack in enumerate(stacks):
        if index == source or len(stack) == max_tiers:
            continue
        if best is None:
            best = index
        elif earliest[index] > blocker:
            if earliest[best] < blocker or earliest[index] < earliest[best]:
                best = index
        elif earliest[index] < blocker and earliest[best] < blocker and earliest[index] > earliest[best]:
            best = index
    return best
