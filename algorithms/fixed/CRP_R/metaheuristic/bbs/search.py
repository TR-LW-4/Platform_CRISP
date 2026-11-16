"""Bounded beam search of Bacci, Mattia and Ventura (2019)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Sequence

from algorithms.fixed.CRP_R.LB.lb1 import lb1
from algorithms.fixed.CRP_R.LB.lb4 import lb4


class Move:
    def __init__(self, container, source, destination):
        self.container = container
        self.source = source
        self.destination = destination


def beam_width(n_total: int) -> int:
    if n_total < 40:
        return 800
    if n_total < 60:
        return 500
    if n_total < 80:
        return 300
    if n_total < 100:
        return 200
    if n_total < 120:
        return 100
    return 50


def chain_upper(stacks, max_tiers, best_ub, choose_destination) -> int:
    state = [list(stack) for stack in stacks]
    priorities = [priority for stack in state for priority in stack]
    if not priorities:
        return 0
    max_priority = max(priorities)
    moved = 0
    target = min(priorities)
    lower_bound = lb1(state, max_tiers)
    while target <= max_priority and lower_bound > 0:
        if moved + lower_bound >= best_ub:
            return max_priority * max_priority
        source = next(index for index, stack in enumerate(state) if target in stack)
        while state[source][-1] != target:
            destination = choose_destination(state, max_tiers)
            if destination is None:
                return max_priority * max_priority
            container = state[source].pop()
            state[destination].append(container)
            moved += 1
            lower_bound = lb1(state, max_tiers)
        state[source].pop()
        target += 1
    return moved


@dataclass
class _Node:
    stacks: List[List[int]]
    cost: int
    target: int
    prefix: List[int] = field(default_factory=list)


def _copy(stacks: Sequence[Sequence[int]]) -> List[List[int]]:
    return [list(stack) for stack in stacks]


def _retrieve(stacks: Sequence[Sequence[int]], target: int, n_total: int):
    state = _copy(stacks)
    blocker = None
    while target <= n_total:
        source = next((index for index, stack in enumerate(state) if target in stack), -1)
        if source < 0 or state[source][-1] != target:
            if source >= 0:
                blocker = state[source][-1]
            break
        state[source].pop()
        target += 1
    return state, target, blocker


def _retrieve_moves(stacks: List[List[int]], moves: List[Move]) -> None:
    while True:
        priorities = [priority for stack in stacks for priority in stack]
        if not priorities:
            return
        target = min(priorities)
        source = next(index for index, stack in enumerate(stacks) if target in stack)
        if stacks[source][-1] != target:
            return
        print(stacks)
        moves.append(Move(target, (source, len(stacks[source])), None))
        stacks[source].pop()


def _apply(stacks: List[List[int]], destination: int, moves: List[Move]) -> None:
    print(stacks)
    priorities = [priority for stack in stacks for priority in stack]
    target = min(priorities)
    source = next(index for index, stack in enumerate(stacks) if target in stack)
    container = stacks[source][-1]
    moves.append(Move(
        container,
        (source, len(stacks[source])),
        (destination, len(stacks[destination]) + 1),
    ))
    stacks[source].pop()
    stacks[destination].append(container)


def _reconstruct(stacks_init, prefix, incumbent, max_tiers, choose_destination) -> List[Move]:
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


def search(stacks_init, n_total, max_tiers, time_limit, choose_destination) -> List[Move]:
    started = time.perf_counter()
    root_stacks = _copy(stacks_init)
    incumbent = chain_upper(root_stacks, max_tiers, n_total * n_total, choose_destination)
    best_prefix: List[int] = []
    beam = [_Node(root_stacks, 0, 1)]
    width = beam_width(n_total)

    def expired() -> bool:
        return time.perf_counter() - started >= time_limit

    while beam and not expired():
        children = []
        for node in beam:
            if expired():
                break
            state, target, blocker = _retrieve(node.stacks, node.target, n_total)
            if target >= n_total or blocker is None:
                continue
            bound = lb4(state, max_tiers) + node.cost
            if bound >= incumbent:
                continue
            source = next(index for index, stack in enumerate(state) if blocker in stack)
            opened = _copy(state)
            opened[source].pop()
            seen_empty = False
            for index in range(len(opened)):
                if index == source or len(opened[index]) >= max_tiers:
                    continue
                if not opened[index]:
                    if seen_empty:
                        continue
                    seen_empty = True
                if expired():
                    break
                child_stacks = _copy(opened)
                child_stacks[index].append(blocker)
                remaining = chain_upper(child_stacks, max_tiers, incumbent, choose_destination)
                child_cost = node.cost + 1 + remaining
                prefix = node.prefix + [index]
                if child_cost < incumbent:
                    incumbent = child_cost
                    best_prefix = prefix
                children.append((
                    child_cost,
                    bound,
                    _Node(child_stacks, node.cost + 1, target, prefix),
                ))
        if not children or expired():
            break
        children.sort(key=lambda item: (item[0], item[1]))
        beam = [item[2] for item in children[:width]]

    return _reconstruct(stacks_init, best_prefix, incumbent, max_tiers, choose_destination)