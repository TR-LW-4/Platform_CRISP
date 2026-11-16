"""
------------------------------- Reference --------------------------------
R. Jovanović, S. Voß,
"A chain heuristic for the Blocks Relocation Problem",
Computers & Industrial Engineering 75 (2014) 79–86.
The destination rule is the ChainF variant used by Bacci, Mattia and
Ventura, Computers & Operations Research 103 (2019) 252–264.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Optional, Sequence


def _scan(stacks: Sequence[Sequence[int]], max_tiers: int, priority: int, max_priority: int, excluded: set):
    best_bad = best_bad_room = -1
    best_good = worst_bad = max_priority * max_priority
    bad = bad_room = worst = good = -1
    for index, stack in enumerate(stacks):
        if index in excluded or len(stack) >= max_tiers:
            continue
        earliest = min(stack) if stack else max_priority + 1
        free = max_tiers - len(stack)
        if earliest < priority:
            if earliest > best_bad:
                best_bad = earliest
                bad = index
            if free > 1 and earliest > best_bad_room:
                best_bad_room = earliest
                bad_room = index
            if earliest < worst_bad:
                worst_bad = earliest
                worst = index
        elif earliest < best_good:
            best_good = earliest
            good = index
    return bad, good, bad_room, worst


def _bad_stack(stacks: Sequence[Sequence[int]], max_tiers: int, bad: int, bad_room: int, worst: int) -> int:
    chosen = bad
    if max_tiers - len(stacks[chosen]) == 1 and bad_room != -1:
        chosen = bad_room
    if max_tiers - len(stacks[chosen]) == 1:
        chosen = worst
    return chosen


def choose_destination(stacks, max_tiers) -> Optional[int]:
    """Destination of the current top blocker under Bacci's ChainF rule."""
    priorities = [priority for stack in stacks for priority in stack]
    if not priorities:
        return None
    max_priority = max(priorities)
    target = min(priorities)
    source = next(index for index, stack in enumerate(stacks) if target in stack)
    if stacks[source][-1] == target:
        return None

    blocker = stacks[source][-1]
    above = len(stacks[source]) - 1 - stacks[source].index(target)
    if above >= 2:
        next_blocker = stacks[source][-2]
    else:
        next_blocker = -1

    bad, good, bad_room, worst = _scan(stacks, max_tiers, blocker, max_priority, {source})
    if next_blocker != -1 and next_blocker > blocker:
        _next_bad, next_good, _next_room, _next_worst = _scan(
            stacks, max_tiers, next_blocker, max_priority, {source}
        )
    else:
        next_good = -1

    if good != -1:
        chosen = good
        if next_blocker != -1 and next_blocker > blocker and next_good != -1 and chosen == next_good:
            bad, good, bad_room, worst = _scan(
                stacks, max_tiers, blocker, max_priority, {source, next_good}
            )
            if good != -1:
                chosen = good
            elif bad != -1:
                chosen = _bad_stack(stacks, max_tiers, bad, bad_room, worst)
        return chosen
    if bad == -1:
        return None
    return _bad_stack(stacks, max_tiers, bad, bad_room, worst)