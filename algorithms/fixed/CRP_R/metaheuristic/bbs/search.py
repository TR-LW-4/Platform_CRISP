"""Bounded beam search of Bacci, Mattia and Ventura (2019)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from algorithms.CRP_R.operator.chain_f import choose_destination

from .lower_bound import blocking_count, ubalb

Move = Tuple[int, int, Optional[int]]


def beam_width(n: int) -> int:
    if n < 40:
        return 800
    if n < 60:
        return 500
    if n < 80:
        return 300
    if n < 100:
        return 200
    if n < 120:
        return 100
    return 50


def chain_upper(stacks: Sequence[Sequence[int]], max_tiers: int, best_ub: int, blocking: int) -> int:
    """Walk ChainF until the bay is clear. Stop early when the partial cost cannot improve."""
    state = _copy(stacks)
    present = [item for stack in state for item in stack]
    if not present:
        return 0
    n = max(present)
    moved = 0
    target = min(present)
    while target <= n and blocking > 0:
        if moved + blocking >= best_ub:
            return n * n
        source = next(index for index, stack in enumerate(state) if target in stack)
        while state[source][-1] != target:
            destination = choose_destination(state, max_tiers)
            if destination is None:
                return n * n
            container = state[source].pop()
            earliest = min(state[destination]) if state[destination] else n + 1
            if earliest > container:
                blocking -= 1
            state[destination].append(container)
            moved += 1
        state[source].pop()
        target += 1
    return moved


@dataclass
class _Node:
    stacks: List[List[int]]
    rs: int
    blocking: int
    min_item: int
    moves: List[int] = field(default_factory=list)


def _copy(stacks: Sequence[Sequence[int]]) -> List[List[int]]:
    return [list(stack) for stack in stacks]


def _retrieve(stacks: Sequence[Sequence[int]], min_item: int, n: int):
    state = _copy(stacks)
    blocker = None
    while min_item <= n:
        source = next((index for index, stack in enumerate(state) if min_item in stack), -1)
        if source < 0 or state[source][-1] != min_item:
            if source >= 0:
                blocker = state[source][-1]
            break
        state[source].pop()
        min_item += 1
    return state, min_item, blocker


def _retrieve_moves(stacks: List[List[int]], moves: List[Move]) -> None:
    while True:
        present = [item for stack in stacks for item in stack]
        if not present:
            return
        target = min(present)
        source = next(index for index, stack in enumerate(stacks) if target in stack)
        if stacks[source][-1] != target:
            return
        moves.append((target, source, None))
        stacks[source].pop()


def _apply(stacks: List[List[int]], destination: int, moves: List[Move]) -> None:
    present = [item for stack in stacks for item in stack]
    target = min(present)
    source = next(index for index, stack in enumerate(stacks) if target in stack)
    container = stacks[source][-1]
    moves.append((container, source, destination))
    stacks[source].pop()
    stacks[destination].append(container)


def _reconstruct(
    stacks_init: Sequence[Sequence[int]],
    prefix: Sequence[int],
    incumbent: int,
    max_tiers: int,
) -> List[Move]:
    stacks = _copy(stacks_init)
    moves: List[Move] = []
    relocations = 0
    for destination in prefix:
        _retrieve_moves(stacks, moves)
        _apply(stacks, destination, moves)
        relocations += 1
    while relocations < incumbent:
        _retrieve_moves(stacks, moves)
        destination = choose_destination(stacks, max_tiers)
        if destination is None:
            break
        _apply(stacks, destination, moves)
        relocations += 1
    _retrieve_moves(stacks, moves)
    return moves


def search(
    stacks_init: Sequence[Sequence[int]],
    n: int,
    max_tiers: int,
    time_limit: float,
    stop: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[float, int], None]] = None,
) -> Tuple[List[Move], int]:
    """Return a relocation plan and its relocation count."""
    started = time.perf_counter()
    root_stacks = _copy(stacks_init)
    root_blocking = blocking_count(root_stacks)
    incumbent = chain_upper(root_stacks, max_tiers, n * n, root_blocking)
    best_prefix: List[int] = []
    beam = [_Node(root_stacks, 0, root_blocking, 1)]
    width = beam_width(n)

    def expired() -> bool:
        if stop is not None and stop():
            return True
        return time.perf_counter() - started >= time_limit

    while beam and not expired():
        children = []
        for node in beam:
            if expired():
                break
            state, min_item, blocker = _retrieve(node.stacks, node.min_item, n)
            if min_item >= n or blocker is None:
                continue
            blocking = node.blocking if node.blocking > 0 else 0
            lower = blocking + ubalb(state, max_tiers, n, min_item) + node.rs
            if lower >= incumbent:
                continue
            source = next(index for index, stack in enumerate(state) if blocker in stack)
            opened = _copy(state)
            opened[source].pop()
            seen_empty = False
            for destination in range(len(opened)):
                if destination == source or len(opened[destination]) >= max_tiers:
                    continue
                if not opened[destination]:
                    if seen_empty:
                        continue
                    seen_empty = True
                if expired():
                    break
                earliest = min(opened[destination]) if opened[destination] else n + 1
                child_blocking = blocking - 1 if earliest > blocker else blocking
                child_stacks = _copy(opened)
                child_stacks[destination].append(blocker)
                remaining = chain_upper(child_stacks, max_tiers, incumbent, child_blocking)
                child_cost = node.rs + 1 + remaining
                prefix = node.moves + [destination]
                if child_cost < incumbent:
                    incumbent = child_cost
                    best_prefix = prefix
                    if on_progress is not None:
                        on_progress(time.perf_counter() - started, incumbent)
                children.append((
                    child_cost,
                    lower,
                    _Node(child_stacks, node.rs + 1, child_blocking, min_item, prefix),
                ))
        if not children or expired():
            break
        children.sort(key=lambda item: (item[0], item[1]))
        beam = [item[2] for item in children[:width]]

    plan = _reconstruct(stacks_init, best_prefix, incumbent, max_tiers)
    return plan, incumbent
