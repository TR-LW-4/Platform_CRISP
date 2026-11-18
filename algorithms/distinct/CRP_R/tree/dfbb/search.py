"""Depth-first branch-and-bound for Kim and Hong (2006)."""

from __future__ import annotations

from typing import List, Sequence

from algorithms.distinct.CRP_R.LB.lb1 import lb1


class Move:
    def __init__(self, container, source, destination):
        self.container = container
        self.source = source
        self.destination = destination

def search(
    stacks_init: Sequence[Sequence[int]],
    max_tiers: int,
) -> List[Move]:
    """Return the optimal relocation plan."""
    best_cost = float("inf")
    best_moves: List[Move] = []

    def improve(cost: float, moves: List[Move]) -> None:
        nonlocal best_cost, best_moves
        if cost >= best_cost:
            return
        best_cost = cost
        best_moves = moves

    def dfs(state: List[List[int]], cost: int, moves: List[Move]) -> None:
        state = _copy(state)
        moves = moves + _retrieve(state)
        if not _any(state):
            improve(cost, moves)
            return
        if cost + lb1(state, max_tiers) >= best_cost:
            return
        children = []
        for child, extra in _assignments(state, max_tiers):
            added = _relocations(extra)
            child_cost = cost + added
            bound = child_cost + lb1(child, max_tiers)
            if bound < best_cost:
                children.append((bound, child, moves + extra, child_cost))
        children.sort(key=lambda item: item[0])
        for _, child, child_moves, child_cost in children:
            dfs(child, child_cost, child_moves)

    dfs(_copy(stacks_init), 0, [])
    return best_moves

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
            container = child[source][-1]
            landing = len(child[destination]) + 1
            step = [Move(
                container,
                (source, len(child[source])),
                (destination, landing),
            )]
            child[source].pop()
            child[destination].append(container)
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
        moves.append(Move(target, (source, len(stacks[source])), None))
        stacks[source].pop()
    return moves


def _destinations(stacks: Sequence[Sequence[int]], source: int, max_tiers: int) -> List[int]:
    return [
        index
        for index, stack in enumerate(stacks)
        if index != source and len(stack) < max_tiers
    ]


def _relocations(moves: Sequence[Move]) -> int:
    return sum(step.destination is not None for step in moves)


def _any(stacks: Sequence[Sequence[int]]) -> bool:
    return any(stacks)


def _copy(stacks: Sequence[Sequence[int]]) -> List[List[int]]:
    return [list(stack) for stack in stacks]