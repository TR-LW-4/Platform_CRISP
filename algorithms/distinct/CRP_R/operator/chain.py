"""
------------------------------- Reference --------------------------------
R. Jovanović, S. Voß,
"A chain heuristic for the Blocks Relocation Problem",
Computers & Industrial Engineering 75 (2014) 79–86.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from typing import Optional


def choose_destination(stacks, max_tiers, chain_f=True) -> Optional[int]:
    priorities = [priority for stack in stacks for priority in stack]
    if not priorities:
        return None

    target = min(priorities)
    source = next(index for index, stack in enumerate(stacks) if target in stack)
    blocker = stacks[source][-1]
    if blocker == target:
        return None

    max_priority = max(priorities)

    def earliest_of(stack):
        return min(stack) if stack else max_priority + 1

    def min_max(priority, excluded, heights=None, overrides=None):
        def height(index):
            if heights is None:
                return len(stacks[index])
            return heights[index]

        def earliest(index):
            if overrides and index in overrides:
                return overrides[index]
            return earliest_of(stacks[index])

        candidates = [
            index for index in range(len(stacks))
            if index not in excluded and height(index) < max_tiers
        ]
        if not candidates:
            return None, 0

        good = [index for index in candidates if earliest(index) > priority]
        if good:
            chosen = min(good, key=lambda index: (earliest(index), index))
        else:
            def bad_key(index):
                stack_earliest = earliest(index)
                if chain_f and height(index) == max_tiers - 1:
                    return (-max_priority - stack_earliest, -index)
                return (stack_earliest, -index)

            chosen = max(candidates, key=bad_key)
        return chosen, earliest(chosen) - priority

    chosen, gap = min_max(blocker, {source})
    if chosen is None:
        return None

    above = stacks[source][stacks[source].index(target) + 1:]
    if len(above) >= 2:
        next_blocker = above[-2]
        next_source = source
    else:
        later = [priority for priority in priorities if priority > target]
        if not later:
            return chosen
        next_target = min(later)
        next_source = next(index for index, stack in enumerate(stacks) if next_target in stack)
        if stacks[next_source][-1] == next_target:
            return chosen
        next_blocker = stacks[next_source][-1]

    if blocker >= next_blocker:
        return chosen

    reserved, next_gap = min_max(next_blocker, {next_source})
    if reserved is None:
        return chosen

    alternate, alternate_gap = min_max(blocker, {source, reserved})
    if alternate is None:
        return chosen

    heights = [len(stack) for stack in stacks]
    heights[source] -= 1
    heights[chosen] += 1
    overrides = {
        source: earliest_of(stacks[source][:-1]),
        chosen: min(earliest_of(stacks[chosen]), blocker),
    }
    _, later_gap = min_max(next_blocker, {next_source}, heights, overrides)

    if gap > 0 and next_gap > 0 and next_gap < gap and alternate_gap * later_gap > 0:
        return alternate
    return chosen