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


def _scan(stacks: Sequence[Sequence[int]], max_tiers: int, item: int, n: int, excluded: set):
    best_bad = best_bad_room = -1
    best_good = worst_bad = n * n
    bad = bad_room = worst = good = -1
    for index, stack in enumerate(stacks):
        if index in excluded or len(stack) >= max_tiers:
            continue
        earliest = min(stack) if stack else n + 1
        free = max_tiers - len(stack)
        if earliest < item:
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
    present = [item for stack in stacks for item in stack]
    if not present:
        return None
    n = max(present)
    target = min(present)
    source = next(index for index, stack in enumerate(stacks) if target in stack)
    if stacks[source][-1] == target:
        return None

    blocker = stacks[source][-1]
    above = len(stacks[source]) - 1 - stacks[source].index(target)
    if above >= 2:
        nxt = stacks[source][-2]
    else:
        nxt = -1

    bad, good, bad_room, worst = _scan(stacks, max_tiers, blocker, n, {source})
    if nxt != -1 and nxt > blocker:
        _bad_n, good_n, _room_n, _worst_n = _scan(stacks, max_tiers, nxt, n, {source})
    else:
        good_n = -1

    if good != -1:
        chosen = good
        if nxt != -1 and nxt > blocker and good_n != -1 and chosen == good_n:
            bad, good, bad_room, worst = _scan(stacks, max_tiers, blocker, n, {source, good_n})
            if good != -1:
                chosen = good
            elif bad != -1:
                chosen = _bad_stack(stacks, max_tiers, bad, bad_room, worst)
        return chosen
    if bad == -1:
        return None
    return _bad_stack(stacks, max_tiers, bad, bad_room, worst)
