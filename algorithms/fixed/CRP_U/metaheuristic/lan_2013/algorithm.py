"""
LAN
<2013> <heuristic> <unrestricted> <single-bay> <CRP-U>
Look-ahead N. N=1 relocates only the current blocker.
N --- 1 --- Number of upcoming containers considered

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

from algorithms.ALGORITHM import Algorithm
from algorithms.fixed.CRP_U.move import Move


class LAN(Algorithm):

    def main(self):
        horizon = self.parameter("N", 1, int)
        stacks = [list(stack) for stack in self.problem.stacks]
        return plan(stacks, self.problem.max_tiers, horizon)


def plan(stacks, max_tiers, horizon):
    state = [list(stack) for stack in stacks]
    remaining = sorted(priority for stack in state for priority in stack)
    moves = []
    limit = max(len(remaining) * (len(remaining) + 1), 1)
    while remaining and len(moves) < limit:
        while remaining and _top(state, remaining[0]) is not None:
            source = _top(state, remaining[0])
            moves.append(Move(remaining.pop(0), source, None))
            state[source].pop()
        if not remaining:
            break
        target = remaining[0]
        home = _find(state, target)
        count = min(horizon, len(remaining))
        count = _shrink(state, remaining, count, max_tiers)
        tops = []
        for priority in remaining[:count]:
            source = _find(state, priority)
            if state[source]:
                tops.append((source, state[source][-1]))
        tops.sort(key=lambda item: -item[1])
        moved = False
        for source, container in tops:
            if source == home or (_good(state, container, max_tiers, source) and container != _earliest(state[source])):
                destination = _choose(state, container, max_tiers, source)
                if destination is None:
                    continue
                state[source].pop()
                state[destination].append(container)
                moves.append(Move(container, source, destination))
                moved = True
                break
        if not moved:
            container = state[home][-1]
            destination = _choose(state, container, max_tiers, home)
            if destination is None:
                break
            state[home].pop()
            state[destination].append(container)
            moves.append(Move(container, home, destination))
    return moves


def _top(state, priority):
    for index, stack in enumerate(state):
        if stack and stack[-1] == priority:
            return index
    return None


def _find(state, priority):
    return next(index for index, stack in enumerate(state) if priority in stack)


def _earliest(stack):
    return min(stack) if stack else float("inf")


def _good(state, container, max_tiers, source):
    return [
        index for index, stack in enumerate(state)
        if index != source and stack and len(stack) < max_tiers and _earliest(stack) > container
    ]


def _choose(state, container, max_tiers, source):
    good = _good(state, container, max_tiers, source)
    if good:
        return min(good, key=lambda index: _earliest(state[index]))
    candidates = [
        index for index, stack in enumerate(state)
        if index != source and len(stack) < max_tiers
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda index: _earliest(state[index]))


def _shrink(state, remaining, count, max_tiers):
    width = len(state)
    while count > 1:
        used = {_find(state, priority) for priority in remaining[:count]}
        outside = [index for index in range(width) if index not in used]
        if len(used) == width or (outside and all(len(state[index]) >= max_tiers for index in outside)):
            count -= 1
        else:
            break
    return max(count, 1)
