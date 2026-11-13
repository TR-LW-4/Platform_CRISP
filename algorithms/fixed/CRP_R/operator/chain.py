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

def choose_destination(stacks, max_tiers, chain_f=True):
    priorities = [p for stack in stacks for p in stack]
    if not priorities:
        return None

    target = min(priorities)
    src = next(i for i, stack in enumerate(stacks) if target in stack)
    blocker = stacks[src][-1]
    if blocker == target:
        return None

    initial_count = max(priorities)

    def earliest_of(stack):
        return min(stack) if stack else initial_count + 1

    def min_max(priority, excluded, heights=None, overrides=None):
        def height(i):
            if heights is None:
                return len(stacks[i])
            return heights[i]

        def earliest(i):
            if overrides and i in overrides:
                return overrides[i]
            return earliest_of(stacks[i])

        usable = [
            i for i in range(len(stacks))
            if i not in excluded and height(i) < max_tiers
        ]
        if not usable:
            return None, 0

        good = [i for i in usable if earliest(i) > priority]
        if good:
            chosen = min(good, key=lambda i: (earliest(i), i))
        else:
            def bad_key(i):
                p = earliest(i)
                if chain_f and height(i) == max_tiers - 1:
                    return (-initial_count - p, -i)
                return (p, -i)

            chosen = max(usable, key=bad_key)
        return chosen, earliest(chosen) - priority

    chosen, gap = min_max(blocker, {src})
    if chosen is None:
        return None

    above = stacks[src][stacks[src].index(target) + 1:]
    if len(above) >= 2:
        nxt = above[-2]
        nxt_src = src
    else:
        later = [p for p in priorities if p > target]
        if not later:
            return chosen
        nxt_target = min(later)
        nxt_src = next(i for i, stack in enumerate(stacks) if nxt_target in stack)
        if stacks[nxt_src][-1] == nxt_target:
            return chosen
        nxt = stacks[nxt_src][-1]

    if blocker >= nxt:
        return chosen

    reserved, next_gap = min_max(nxt, {nxt_src})
    if reserved is None:
        return chosen

    alternate, alternate_gap = min_max(blocker, {src, reserved})
    if alternate is None:
        return chosen

    heights = [len(stack) for stack in stacks]
    heights[src] -= 1
    heights[chosen] += 1
    overrides = {
        src: earliest_of(stacks[src][:-1]),
        chosen: min(earliest_of(stacks[chosen]), blocker),
    }
    _, later_gap = min_max(nxt, {nxt_src}, heights, overrides)

    if gap > 0 and next_gap > 0 and next_gap < gap and alternate_gap * later_gap > 0:
        return alternate
    return chosen