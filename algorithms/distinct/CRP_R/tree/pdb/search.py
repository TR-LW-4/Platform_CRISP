"""Depth-first branch-and-bound with a pattern database for Ku and Arthanari (2016)."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

from algorithms.distinct.CRP_R.LB.lb3 import lb3
from algorithms.distinct.CRP_R.operator import min_max, pr1, pr2, pr4

from .levels import abstract_key

Move = Tuple[int, int, Optional[int]]
_PROBES = (pr1, pr2, min_max, pr4)


def search(
    stacks_init: Sequence[Sequence[int]],
    max_tiers: int,
    levels: Dict,
    cache_level: int = 15,
    max_cache_size: int = 20_000_000,
    time_limit: float = 0.0,
) -> Tuple[List[Move], Optional[int]]:
    """Return the best relocation plan found and its relocation count."""
    deadline = None if time_limit <= 0 else time.perf_counter() + time_limit
    root = _copy(stacks_init)
    prefix = _retrieve(root)
    if not _any(root):
        return prefix, 0

    probe_moves, best_cost = _probe(root, max_tiers)
    best_moves: List[Move] = prefix + probe_moves
    cache: Dict = {}
    stopped = False

    def improve(cost: float, moves: List[Move]) -> None:
        nonlocal best_cost, best_moves
        if cost < best_cost:
            best_cost = cost
            best_moves = moves

    def dfs(state, cost, moves, level) -> None:
        nonlocal stopped
        if stopped:
            return
        if deadline is not None and time.perf_counter() > deadline:
            stopped = True
            return
        if not _any(state):
            improve(cost, moves)
            return
        if cost + lb3(state, max_tiers) >= best_cost:
            return

        table = levels.get(sum(len(stack) for stack in state))
        if table is not None:
            rest = table[abstract_key(state)]
            if cost + rest < best_cost:
                improve(cost + rest, moves + _complete(state, levels, max_tiers))
            return

        if level <= cache_level:
            key = abstract_key(state)
            seen = cache.get(key)
            if seen is not None and seen <= cost:
                return
            if seen is not None or len(cache) < max_cache_size:
                cache[key] = cost

        children = []
        for child, extra in _assignments(state, max_tiers):
            child_cost = cost + _relocations(extra)
            bound = child_cost + lb3(child, max_tiers)
            if bound < best_cost:
                children.append((bound, child, moves + extra, child_cost))
        children.sort(key=lambda item: item[0])
        for bound, child, child_moves, child_cost in children:
            if bound < best_cost:
                dfs(child, child_cost, child_moves, level + 1)

    dfs(root, 0, prefix, 0)
    if best_cost == float("inf"):
        return [], None
    return best_moves, int(best_cost)


def _probe(stacks, max_tiers):
    best_moves: List[Move] = []
    best_cost = float("inf")
    for probe in _PROBES:
        moves, cost = _rollout(stacks, max_tiers, probe.choose_destination)
        if cost < best_cost:
            best_moves = moves
            best_cost = cost
    return best_moves, best_cost


def _rollout(stacks, max_tiers, choose):
    state = _copy(stacks)
    moves: List[Move] = []
    cost = 0
    while True:
        moves.extend(_retrieve(state))
        if not _any(state):
            return moves, cost
        destination = choose(state, max_tiers)
        if destination is None:
            return moves, float("inf")
        target = min(item for stack in state for item in stack)
        source = next(index for index, stack in enumerate(state) if target in stack)
        container = state[source].pop()
        state[destination].append(container)
        moves.append((container, source, destination))
        cost += 1


def _assignments(stacks, max_tiers):
    target = min(item for stack in stacks for item in stack)
    source = next(index for index, stack in enumerate(stacks) if target in stack)
    above = len(stacks[source]) - stacks[source].index(target) - 1
    found = []

    def walk(state, moves, left):
        if left == 0:
            found.append((state, moves))
            return
        for destination in _destinations(state, source, max_tiers):
            child = _copy(state)
            container = child[source].pop()
            child[destination].append(container)
            step = [(container, source, destination)]
            if left == 1:
                step.extend(_retrieve(child))
            walk(child, moves + step, left - 1)

    walk(_copy(stacks), [], above)
    return found


def _complete(stacks, levels, max_tiers) -> List[Move]:
    state = _copy(stacks)
    moves: List[Move] = []
    while _any(state):
        value = _exact(state, levels)
        target = min(item for stack in state for item in stack)
        source = next(index for index, stack in enumerate(state) if target in stack)
        for destination in _destinations(state, source, max_tiers):
            child = _copy(state)
            container = child[source].pop()
            child[destination].append(container)
            extra = _retrieve(child)
            if 1 + _exact(child, levels) == value:
                moves.append((container, source, destination))
                moves.extend(extra)
                state = child
                break
        else:
            raise RuntimeError("pattern database has no optimal move")
    return moves


def _exact(stacks, levels) -> float:
    remaining = sum(len(stack) for stack in stacks)
    return levels[remaining][abstract_key(stacks)] if remaining else 0


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


def _relocations(moves: Sequence[Move]) -> int:
    return sum(step[2] is not None for step in moves)


def _any(stacks) -> bool:
    return any(stacks)


def _copy(stacks) -> List[List[int]]:
    return [list(stack) for stack in stacks]