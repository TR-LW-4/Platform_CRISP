"""Depth-first branch-and-bound for Kim and Hong (2006)."""

from __future__ import annotations

import time
from typing import Callable, List, Optional, Sequence, Tuple

from algorithms.CRP_R.bound.lb1 import lb1

Move = Tuple[int, int, Optional[int]]


def search(
    stacks_init: Sequence[Sequence[int]],
    max_tiers: int,
    time_limit: float,
    stop: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[float, int], None]] = None,
) -> Tuple[List[Move], Optional[int]]:
    """Return the relocation plan and its relocation count."""
    started = time.perf_counter()
    best_cost = float("inf")
    best_moves: List[Move] = []

    def expired() -> bool:
        if stop is not None and stop():
            return True
        return time.perf_counter() - started >= time_limit

    def improve(cost: float, moves: List[Move]) -> None:
        nonlocal best_cost, best_moves
        if cost >= best_cost:
            return
        best_cost = cost
        best_moves = moves
        if on_progress is not None:
            on_progress(time.perf_counter() - started, int(cost))

    def dfs(state: List[List[int]], cost: int, moves: List[Move]) -> None:
        if expired():
            return
        state = _copy(state)
        moves = moves + _retrieve(state)
        if not _any(state):
            improve(cost, moves)
            return
        if cost + lb1(state) >= best_cost:
            return
        children = []
        for child, extra in _assignments(state, max_tiers):
            added = _relocations(extra)
            child_cost = cost + added
            bound = child_cost + lb1(child)
            if bound < best_cost:
                children.append((bound, child, moves + extra, child_cost))
        children.sort(key=lambda item: item[0])
        for _, child, child_moves, child_cost in children:
            if expired():
                return
            dfs(child, child_cost, child_moves)

    root = _copy(stacks_init)
    dfs(root, 0, [])
    if best_cost == float("inf"):
        return [], None
    return best_moves, int(best_cost)


def _assignments(stacks: Sequence[Sequence[int]], max_tiers: int):
    present = [item for stack in stacks for item in stack]
    if not present:
        return []
    target = min(present)
    source = next(index for index, stack in enumerate(stacks) if target in stack)
    above = len(stacks[source]) - stacks[source].index(target) - 1
    if above <= 0:
        return []
    found = []

    def walk(state, moves, left):
        if left == 0:
            found.append((_copy(state), list(moves)))
            return
        for destination in _destinations(state, source, max_tiers):
            child = _copy(state)
            container = child[source].pop()
            child[destination].append(container)
            step = [(container, source, destination)]
            step.extend(_retrieve(child))
            walk(child, moves + step, left - 1)

    walk(_copy(stacks), [], above)
    return found


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


def _destinations(stacks: Sequence[Sequence[int]], source: int, max_tiers: int) -> List[int]:
    return [
        index
        for index, stack in enumerate(stacks)
        if index != source and len(stack) < max_tiers
    ]


def _relocations(moves: Sequence[Move]) -> int:
    return sum(destination is not None for _, _, destination in moves)


def _any(stacks: Sequence[Sequence[int]]) -> bool:
    return any(stacks)


def _copy(stacks: Sequence[Sequence[int]]) -> List[List[int]]:
    return [list(stack) for stack in stacks]
