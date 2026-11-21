"""
Greedy heuristic for the block relocation problem with a stowage plan.
<2019> <heuristic> <restricted> <single-bay> <CRP-Stow-R>

------------------------------- Reference --------------------------------
S. Tanaka, S. Voß,
"An exact algorithm for the block relocation problem with a stowage plan",
European Journal of Operational Research 279 (2019) 767–781. Section 6.3.
The constructed sequence only relocates containers above the chosen target,
so it is feasible for the restricted problem and a feasible upper bound for
the unrestricted problem.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from algorithms.ALGORITHM import Algorithm
from algorithms.stowage.CRP_S_U.LB.lb2c4c import cycle_sets
from algorithms.stowage.CRP_S_U.model import loaded_from, normalize, top_target


_EMPTY = 10 ** 6


def relocations(stacks, max_tiers, heights):
    """Return a feasible restricted relocation list ``[(source, destination), ...]``."""
    state = normalize(stacks, heights)
    moves = []
    while any(state):
        blocking2, blocking4, _ = cycle_sets(state)
        loaded = loaded_from(state, heights)
        source = _choose_source(state, loaded, heights, blocking2, blocking4)
        while True:
            loaded = loaded_from(state, heights)
            target = top_target(state[source], loaded)
            if target is None or target == len(state[source]) - 1:
                break
            destination = _choose_destination(state, source, max_tiers, heights)
            container = state[source].pop()
            state[destination].append(container)
            moves.append((source, destination))
        state = normalize(state, heights)
    return moves


def _choose_source(state, loaded, heights, blocking2, blocking4):
    best_index = None
    best_key = None
    for index, stack in enumerate(state):
        target = top_target(stack, loaded)
        if target is None:
            continue
        stacking = stack[target + 1:]
        if not stacking:
            continue
        below = stack[:target]
        neither = sum(
            1 for container in stacking
            if container not in blocking2 and container not in blocking4
        )
        blocking4_count = sum(1 for container in stacking if container in blocking4)
        blocking2_count = sum(1 for container in stacking if container in blocking2)
        soon = min(_priority(container, loaded) for container in stacking)
        goal = stack[target]
        remaining = heights[goal[0]] - goal[1]
        if below:
            buried = min(_priority(container, loaded) for container in below)
        else:
            buried = _EMPTY
        key = (neither, blocking4_count, blocking2_count, -soon, -remaining, buried, index)
        if best_key is None or key < best_key:
            best_key = key
            best_index = index
    if best_index is None:
        raise ValueError("no target container is left to retrieve")
    return best_index


def _choose_destination(state, source, max_tiers, heights):
    container = state[source][-1]
    loaded = loaded_from(state, heights)
    best_index = None
    best_key = None
    for index, stack in enumerate(state):
        if index == source or len(stack) >= max_tiers:
            continue
        key = _destination_key(state, source, index, container, stack, loaded)
        if best_key is None or key < best_key:
            best_key = key
            best_index = index
    if best_index is None:
        raise ValueError("no stack can accept the relocated container")
    return best_index


def _destination_key(state, source, index, container, stack, loaded):
    if stack:
        top = stack[-1]
        trial = [list(column) for column in state]
        trial[source].pop()
        trial[index].append(container)
        blocking2, blocking4, _ = cycle_sets(trial)
        becomes_blocking = container in blocking2
        becomes_cycle = container in blocking4
        if (
            not becomes_blocking
            and not becomes_cycle
            and container[0] == top[0]
            and container[1] < top[1]
        ):
            return (0, top[1], index)
        earliest = min(_priority(item, loaded) for item in stack)
        return (1, int(becomes_blocking), int(becomes_cycle), -earliest, index)
    return (1, 0, 0, -_EMPTY, index)


def _priority(container, loaded):
    vessel_stack, vessel_tier = container
    return vessel_tier - loaded[vessel_stack]


class TV(Algorithm):
    compatible_problems = ["CRP-Stow-R"]

    def main(self):
        moves = relocations(
            self.problem.stacks,
            self.problem.max_tiers,
            self.problem.vessel_heights,
        )
        return self.problem.replay(moves)
