"""Greedy beam search: one relocation per level, scored by a completion heuristic."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from .scoring import sdh_cost, vrh_cost

Move = Tuple[int, int, Optional[int]]
Score = Callable[[Sequence[Sequence[int]], int, int], int]


@dataclass
class _Node:
    stacks: List[List[int]]
    g: int
    moves: List[Move] = field(default_factory=list)


def _copy(stacks: Sequence[Sequence[int]]) -> List[List[int]]:
    return [list(stack) for stack in stacks]


def _retrieve(stacks: List[List[int]], moves: List[Move]) -> None:
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


def _children(node: _Node, max_tiers: int) -> List[Tuple[_Node, int]]:
    present = [item for stack in node.stacks for item in stack]
    if not present:
        return []
    target = min(present)
    source = next(index for index, stack in enumerate(node.stacks) if target in stack)
    if node.stacks[source][-1] == target:
        return []
    children = []
    for destination in range(len(node.stacks)):
        if destination == source or len(node.stacks[destination]) >= max_tiers:
            continue
        stacks = _copy(node.stacks)
        container = stacks[source].pop()
        stacks[destination].append(container)
        moves = list(node.moves)
        moves.append((container, source, destination))
        _retrieve(stacks, moves)
        children.append((_Node(stacks, node.g + 1, moves), destination))
    return children


def _select(
    children: List[Tuple[_Node, float, int]],
    width: int,
    diverse_parents: bool,
) -> List[_Node]:
    children.sort(key=lambda item: item[1])
    if len(children) <= width:
        return [item[0] for item in children]
    threshold = children[width - 1][1]
    selected = [item for item in children if item[1] < threshold]
    tied = [item for item in children if item[1] == threshold]
    if diverse_parents:
        seen = {item[2] for item in selected}
        fresh = [item for item in tied if item[2] not in seen]
        repeated = [item for item in tied if item[2] in seen]
        tied = fresh + repeated
    selected.extend(tied[: width - len(selected)])
    return [item[0] for item in selected]


def search(
    stacks_init: Sequence[Sequence[int]],
    n: int,
    max_tiers: int,
    time_limit: float,
    width: int = 5,
    evaluator: str = "VRH",
    stop: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[float, int], None]] = None,
) -> Tuple[Optional[List[Move]], Optional[int]]:
    """Return the first completed plan found, or none when the beam stops early."""
    score: Score = vrh_cost if evaluator.upper() == "VRH" else sdh_cost
    started = time.perf_counter()
    root_stacks = _copy(stacks_init)
    root_moves: List[Move] = []
    _retrieve(root_stacks, root_moves)
    if not any(root_stacks):
        return root_moves, 0

    beam = [_Node(root_stacks, 0, root_moves)]
    width = max(1, int(width))

    def expired() -> bool:
        if stop is not None and stop():
            return True
        return time.perf_counter() - started >= time_limit

    while beam:
        if expired():
            return None, None
        children: List[Tuple[_Node, float, int]] = []
        for parent_index, parent in enumerate(beam):
            for child, _destination in _children(parent, max_tiers):
                if not any(child.stacks):
                    if on_progress is not None:
                        on_progress(time.perf_counter() - started, child.g)
                    return child.moves, child.g
                children.append((child, float(score(child.stacks, n, max_tiers)), parent_index))
        if not children:
            break
        beam = _select(children, width, True)
    return None, None
