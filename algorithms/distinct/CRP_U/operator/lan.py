"""
LA-N relocation rule for CRP-U

If look_ahead is not set, it defaults to the number of stacks minus one.

------------------------------- Reference --------------------------------
M.E.H. Petering, M.I. Hussein,
"A new mixed integer program and extended look-ahead heuristic algorithm
 for the block relocation problem",
European Journal of Operational Research 231 (2013) 120–130.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from typing import Optional, Tuple


def choose_move(stacks, max_tiers, look_ahead=None) -> Optional[Tuple[int, int]]:
    max_priority, earliest = _earliest(stacks)
    if max_priority == 0:
        return None
    target = min(earliest)
    home = earliest.index(target)
    if stacks[home][-1] == target:
        return None
    if look_ahead is None:
        look_ahead = len(stacks) - 1
    tops = sorted(((stacks[source][-1], source) for source in _horizon(stacks, max_tiers, look_ahead)),
                  reverse=True)
    for blocker, source in tops:
        if source == home or _cleaning(stacks, max_tiers, earliest, source, blocker):
            destination = _destination(stacks, max_tiers, earliest, source, blocker)
            if destination is not None:
                return source, destination
    return None


def _earliest(stacks):
    max_priority = max((max(stack) for stack in stacks if stack), default=0)
    return max_priority, [min(stack) if stack else max_priority + 1 for stack in stacks]


def _horizon(stacks, max_tiers, look_ahead):
    where = {priority: index for index, stack in enumerate(stacks) for priority in stack}
    priorities = sorted(where)
    count = min(look_ahead, len(priorities))
    while True:
        used = {where[priority] for priority in priorities[:count]}
        if count == 1:
            return used
        outside = [index for index in range(len(stacks)) if index not in used]
        if outside and any(len(stacks[index]) < max_tiers for index in outside):
            return used
        count -= 1


def _cleaning(stacks, max_tiers, earliest, source, blocker):
    if blocker == earliest[source]:
        return False
    return any(earliest[index] > blocker and len(stack) < max_tiers
               for index, stack in enumerate(stacks))


def _destination(stacks, max_tiers, earliest, source, blocker):
    open_stacks = [index for index, stack in enumerate(stacks)
                   if index != source and len(stack) < max_tiers]
    if not open_stacks:
        return None
    good = [index for index in open_stacks if earliest[index] > blocker]
    if good:
        return min(good, key=lambda index: earliest[index])
    return max(open_stacks, key=lambda index: earliest[index])
