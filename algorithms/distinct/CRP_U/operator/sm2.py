"""
SM-2 relocation rule for CRP-U

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
    max_priority, earliest = _earliest(stacks)
    if max_priority == 0:
        return None
    target = min(earliest)
    source = earliest.index(target)
    if stacks[source][-1] == target:
        return None
    move = (_safe_1(stacks, max_tiers, max_priority, earliest)
            or _safe_2(stacks, max_tiers, max_priority, earliest))
    if move is not None:
        return move
    destination = _la_destination(stacks, max_tiers, earliest, source, stacks[source][-1])
    return None if destination is None else (source, destination)


def safe_1(stacks, max_tiers) -> Optional[Tuple[int, int]]:
    max_priority, earliest = _earliest(stacks)
    return _safe_1(stacks, max_tiers, max_priority, earliest)


def safe_2(stacks, max_tiers) -> Optional[Tuple[int, int]]:
    max_priority, earliest = _earliest(stacks)
    return _safe_2(stacks, max_tiers, max_priority, earliest)


def _earliest(stacks):
    max_priority = max((max(stack) for stack in stacks if stack), default=0)
    return max_priority, [min(stack) if stack else max_priority + 1 for stack in stacks]


def _blockers(stacks, earliest):
    return [(index, stack[-1]) for index, stack in enumerate(stacks)
            if len(stack) > 1 and stack[-1] > earliest[index]]


def _safe_1(stacks, max_tiers, max_priority, earliest):
    return _best_safe(stacks, max_tiers, earliest, max_priority, _blockers(stacks, earliest))[0]


def _safe_2(stacks, max_tiers, max_priority, earliest):
    blockers = _blockers(stacks, earliest)
    best, best_gap = None, 4 * max_priority
    for source, stack in enumerate(stacks):
        if not stack:
            continue
        below = min(stack[:-1]) if len(stack) > 1 else max_priority + 1
        relocation = None
        for other, blocker in blockers:
            if other == source or stack[-1] >= blocker or below < blocker:
                continue
            if relocation is None:
                relocation = _best_safe(stacks, max_tiers, earliest, max_priority, [(source, stack[-1])])
            move, gap = relocation
            if move is not None and below - blocker + gap < best_gap:
                best, best_gap = move, below - blocker + gap
    return best


def _best_safe(stacks, max_tiers, earliest, max_priority, blockers):
    best, best_gap = None, max_priority + 1
    for source, blocker in blockers:
        for destination in range(len(stacks)):
            if (destination != source and len(stacks[destination]) < max_tiers
                    and earliest[destination] >= blocker
                    and earliest[destination] - blocker < best_gap):
                best, best_gap = (source, destination), earliest[destination] - blocker
    return best, best_gap


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
