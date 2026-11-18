"""Iterative deepening A* for Zhu et al. (2012)."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from algorithms.distinct.CRP_R.LB.lb1 import lb1
from algorithms.distinct.CRP_R.LB.lb2 import lb2
from algorithms.distinct.CRP_R.LB.lb3 import lb3
from algorithms.distinct.CRP_R.operator import min_max, pr1, pr2, pr4

Move = Tuple[int, int, Optional[int]]
_BOUNDS = ("lb1", "lb2", "lb3")
_PROBES = {"pr1": pr1, "pr2": pr2, "pr3": min_max, "pr4": pr4}


def search(
    stacks_init: Sequence[Sequence[int]],
    max_tiers: int,
    lb_mode: str = "lb3",
    probe_mode: str = "pr4",
) -> Tuple[List[Move], Optional[int]]:
    """Return the optimal relocation plan and its relocation count."""
    mode = lb_mode.lower()
    if mode not in _BOUNDS:
        mode = "lb3"
    probe = probe_mode.lower()
    if probe not in _PROBES:
        probe = "pr4"
    bound_of = {
        "lb1": lambda state: lb1(state, max_tiers),
        "lb2": lambda state: lb2(state, max_tiers),
        "lb3": lambda state: lb3(state, max_tiers),
    }[mode]

    root = _copy(stacks_init)
    prefix = _retrieve(root)
    if not _any(root):
        return prefix, 0

    best_moves, best_cost = _pr_plus(root, max_tiers)
    best_moves = prefix + best_moves

    def improve(cost: float, moves: List[Move]) -> None:
        nonlocal best_cost, best_moves
        if cost >= best_cost:
            return
        best_cost = cost
        best_moves = moves

    def dfs(state, cost, moves, threshold) -> float:
        if not _any(state):
            improve(cost, moves)
            return float("inf")
        node_bound = cost + bound_of(state)
        if node_bound > threshold:
            if node_bound < best_cost:
                rollout_moves, rollout_cost, complete = _rollout(state, max_tiers, probe)
                if complete:
                    improve(cost + rollout_cost, moves + rollout_moves)
            return node_bound
        if node_bound >= best_cost:
            return float("inf")
        next_threshold = float("inf")
        for child, extra in _single_moves(state, max_tiers):
            child_threshold = dfs(child, cost + 1, moves + extra, threshold)
            next_threshold = min(next_threshold, child_threshold)
        return next_threshold

    threshold = float(bound_of(root))
    while True:
        next_threshold = dfs(root, 0, prefix, threshold)
        if best_cost <= threshold:
            break
        if next_threshold == float("inf"):
            break
        threshold = max(threshold + 1.0, next_threshold)

    if best_cost == float("inf"):
        return [], None
    return best_moves, int(best_cost)


def _pr_plus(stacks, max_tiers):
    best_moves: List[Move] = []
    best_cost = float("inf")
    for mode in _PROBES:
        moves, cost, complete = _rollout(stacks, max_tiers, mode)
        if complete and cost < best_cost:
            best_moves = moves
            best_cost = cost
    return best_moves, best_cost


def _rollout(stacks, max_tiers, mode):
    state = _copy(stacks)
    moves: List[Move] = []
    cost = 0
    guard = sum(len(stack) for stack in state) * max(max_tiers, 1) * max(len(state), 1) + 16
    choose = _PROBES[mode].choose_destination
    while _any(state) and cost <= guard:
        moves.extend(_retrieve(state))
        if not _any(state):
            return moves, cost, True
        destination = choose(state, max_tiers)
        if destination is None:
            break
        target = min(item for stack in state for item in stack)
        source = next(index for index, stack in enumerate(state) if target in stack)
        blocker = state[source].pop()
        state[destination].append(blocker)
        moves.append((blocker, source, destination))
        cost += 1
    return moves, cost, not _any(state)


def _single_moves(stacks, max_tiers):
    present = [item for stack in stacks for item in stack]
    target = min(present)
    source = next(index for index, stack in enumerate(stacks) if target in stack)
    blocker = stacks[source][-1]
    children = []
    for destination in _destinations(stacks, source, max_tiers):
        child = _copy(stacks)
        child[source].pop()
        child[destination].append(blocker)
        extra = [(blocker, source, destination)]
        extra.extend(_retrieve(child))
        children.append((child, extra))
    return children


def _retrieve(stacks: List[List[int]]) -> List[Move]:
    moves: List[Move] = []
    while True:
        present = [item for stack in stacks for item in stack]
        if not present:
            return moves
        target = min(present)
        source = next(index for index, stack in enumerate(stacks) if target in stack)
        if stacks[source][-1] != target:
            return moves
        stacks[source].pop()
        moves.append((target, source, None))


def _destinations(stacks, source, max_tiers) -> List[int]:
    return [
        index
        for index, stack in enumerate(stacks)
        if index != source and len(stack) < max_tiers
    ]


def _any(stacks) -> bool:
    return any(stacks)


def _copy(stacks) -> List[List[int]]:
    return [list(stack) for stack in stacks]