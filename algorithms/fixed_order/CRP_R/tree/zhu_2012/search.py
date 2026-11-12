"""Iterative deepening A* for Zhu et al. (2012)."""

from __future__ import annotations

import time
from typing import Callable, List, Optional, Sequence, Tuple

from algorithms.CRP_R.bound.lb1 import lb1
from algorithms.CRP_R.bound.lb2 import lb2
from algorithms.CRP_R.bound.lb3 import lb3

Move = Tuple[int, int, Optional[int]]
_PROBES = ("PR1", "PR2", "PR3", "PR4")


def search(
    stacks_init: Sequence[Sequence[int]],
    n: int,
    max_tiers: int,
    time_limit: float,
    lb_mode: str = "LB3",
    probe_mode: str = "PR4",
    stop: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[float, int], None]] = None,
) -> Tuple[List[Move], Optional[int]]:
    """Return the relocation plan and its relocation count."""
    started = time.perf_counter()
    mode = lb_mode if lb_mode in ("LB1", "LB2", "LB3") else "LB3"
    probe = probe_mode if probe_mode in _PROBES else "PR4"
    bound_of = {
        "LB1": lambda state: lb1(state),
        "LB2": lambda state: lb2(state, max_tiers),
        "LB3": lambda state: lb3(state, max_tiers),
    }[mode]

    root = _copy(stacks_init)
    prefix = _retrieve(root)
    if not _any(root):
        return prefix, 0

    best_moves, best_cost = _pr_plus(root, n, max_tiers)
    best_moves = prefix + best_moves

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

    def dfs(state, cost, moves, threshold) -> float:
        if expired():
            return float("inf")
        if not _any(state):
            improve(cost, moves)
            return float("inf")
        node_bound = cost + bound_of(state)
        if node_bound > threshold:
            if node_bound < best_cost:
                rollout_moves, rollout_cost, complete = _rollout(state, n, max_tiers, probe)
                if complete:
                    improve(cost + rollout_cost, moves + rollout_moves)
            return node_bound
        if node_bound >= best_cost:
            return float("inf")
        next_threshold = float("inf")
        for child, extra in _single_moves(state, max_tiers):
            if expired():
                return float("inf")
            child_threshold = dfs(child, cost + 1, moves + extra, threshold)
            next_threshold = min(next_threshold, child_threshold)
        return next_threshold

    threshold = float(bound_of(root))
    while not expired():
        next_threshold = dfs(root, 0, prefix, threshold)
        if expired():
            break
        if best_cost <= threshold:
            break
        if next_threshold == float("inf"):
            break
        threshold = max(threshold + 1.0, next_threshold)

    if best_cost == float("inf"):
        return [], None
    return best_moves, int(best_cost)


def _pr_plus(stacks, n, max_tiers):
    best_moves: List[Move] = []
    best_cost = float("inf")
    for mode in _PROBES:
        moves, cost, complete = _rollout(stacks, n, max_tiers, mode)
        if complete and cost < best_cost:
            best_moves = moves
            best_cost = cost
    return best_moves, best_cost


def _rollout(stacks, n, max_tiers, mode):
    state = _copy(stacks)
    moves: List[Move] = []
    cost = 0
    guard = sum(len(stack) for stack in state) * max(max_tiers, 1) * max(len(state), 1) + 16
    choose = {"PR1": _pr1, "PR2": _pr2, "PR3": _pr3, "PR4": _pr4}[mode]
    while _any(state) and cost <= guard:
        moves.extend(_retrieve(state))
        if not _any(state):
            return moves, cost, True
        target = min(item for stack in state for item in stack)
        source = next(index for index, stack in enumerate(state) if target in stack)
        blocker = state[source][-1]
        destination = choose(state, source, blocker, max_tiers, n)
        if destination is None:
            break
        state[source].pop()
        state[destination].append(blocker)
        moves.append((blocker, source, destination))
        cost += 1
    return moves, cost, not _any(state)


def _pr1(stacks, source, blocker, max_tiers, n):
    del blocker, n
    destinations = _destinations(stacks, source, max_tiers)
    if not destinations:
        return None
    return min(destinations, key=lambda index: (len(stacks[index]), index))


def _pr2(stacks, source, blocker, max_tiers, n):
    del blocker
    destinations = _destinations(stacks, source, max_tiers)
    if not destinations:
        return None
    return max(destinations, key=lambda index: (_earliest(stacks[index], n), -index))


def _pr3(stacks, source, blocker, max_tiers, n):
    destinations = _destinations(stacks, source, max_tiers)
    if not destinations:
        return None
    safe = [index for index in destinations if _earliest(stacks[index], n) > blocker]
    if safe:
        return min(safe, key=lambda index: _earliest(stacks[index], n))
    return max(destinations, key=lambda index: _earliest(stacks[index], n))


def _pr4(stacks, source, blocker, max_tiers, n):
    destinations = _destinations(stacks, source, max_tiers)
    if not destinations:
        return None
    safe = [index for index in destinations if _earliest(stacks[index], n) > blocker]
    if safe:
        return min(safe, key=lambda index: _earliest(stacks[index], n))
    ranked = sorted(destinations, key=lambda index: -_earliest(stacks[index], n))
    best = ranked[0]
    ordered = sorted(stacks[source])
    second = len(ordered) >= 2 and blocker == ordered[1]
    one_slot = max_tiers - len(stacks[best]) == 1
    if not second and one_slot and len(ranked) > 1:
        return ranked[1]
    return best


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


def _earliest(stack, n) -> int:
    return min(stack) if stack else n + 1


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
