"""Best-first branch-and-bound for Expósito-Izquierdo et al. (2015)."""

from __future__ import annotations

import heapq
import itertools
from typing import List, Optional, Sequence

from algorithms.distinct.CRP_R.LB.lb2 import lb2


class Move:
    def __init__(self, container, source, destination):
        self.container = container
        self.source = source
        self.destination = destination


def search(
    stacks_init: Sequence[Sequence[int]],
    n_total: int,
    max_tiers: int,
    alpha: int = 0,
) -> List[Move]:
    """Return the best relocation plan found in the best-first tree.

    ``alpha`` is the maximum number of assignments kept at a node.
    Zero keeps every assignment.
    """
    limit: Optional[int] = None if alpha <= 0 else alpha
    best_cost = float("inf")
    best_moves: List[Move] = []

    def improve(cost: float, moves: List[Move]) -> None:
        nonlocal best_cost, best_moves
        if cost >= best_cost:
            return
        best_cost = cost
        best_moves = moves

    root = _copy(stacks_init)
    root_moves = _retrieve(root)
    greedy_moves, greedy_cost = _greedy(root, n_total, max_tiers)
    if greedy_cost is not None:
        improve(greedy_cost, root_moves + greedy_moves)
        
    if alpha <= 0:
        seeded = search(stacks_init, n_total, max_tiers, alpha=4)
        improve(_relocations(seeded), seeded)

    if not _any(root):
        return root_moves

    counter = itertools.count()
    queue = [(lb2(root, max_tiers), next(counter), root, 0, root_moves)]
    while queue:
        node_bound, _, state, cost, moves = heapq.heappop(queue)
        if node_bound >= best_cost:
            break
        if not _any(state):
            improve(cost, moves)
            continue
        ranked = _ranked_assignments(state, n_total, max_tiers)
        if limit is not None:
            ranked = ranked[:limit]
        for child, extra in ranked:
            added = _relocations(extra)
            child_cost = cost + added
            child_moves = moves + extra
            if not _any(child):
                improve(child_cost, child_moves)
                continue
            child_bound = child_cost + lb2(child, max_tiers)
            if child_bound < best_cost:
                heapq.heappush(
                    queue,
                    (child_bound, next(counter), child, child_cost, child_moves),
                )

    return best_moves


def _greedy(stacks, n_total, max_tiers):
    state = _copy(stacks)
    moves: List[Move] = []
    cost = 0
    guard = n_total * max(max_tiers, 1) * max(len(state), 1) + 16
    while _any(state) and cost <= guard:
        ranked = _ranked_assignments(state, n_total, max_tiers)
        if not ranked:
            return moves, None
        child, extra = ranked[0]
        moves.extend(extra)
        cost += _relocations(extra)
        state = child
    if _any(state):
        return moves, None
    return moves, cost


def _ranked_assignments(stacks, n_total, max_tiers):
    found = _assignments(stacks, max_tiers)
    found.sort(key=lambda item: _score(stacks, item[1], n_total))
    return found


def _score(stacks, moves, n_total) -> float:
    values = [min(stack) if stack else n_total + 1 for stack in stacks]
    total = sum(
        values[step.destination[0]]
        for step in moves
        if step.destination is not None
    )
    return 1.0 / (1.0 + total)


def _assignments(stacks, max_tiers):
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


def _destinations(stacks, source, max_tiers) -> List[int]:
    return [
        index
        for index, stack in enumerate(stacks)
        if index != source and len(stack) < max_tiers
    ]


def _relocations(moves: Sequence[Move]) -> int:
    return sum(step.destination is not None for step in moves)


def _any(stacks) -> bool:
    return any(stacks)


def _copy(stacks) -> List[List[int]]:
    return [list(stack) for stack in stacks] 