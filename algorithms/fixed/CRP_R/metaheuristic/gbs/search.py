"""Greedy beam search: one relocation per level, scored by a completion heuristic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple


class Move:
    def __init__(self, container, source, destination):
        self.container = container
        self.source = source
        self.destination = destination


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
        moves.append(Move(target, (source, len(stacks[source])), None))
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
        container = stacks[source][-1]
        moves = list(node.moves)
        moves.append(Move(
            container,
            (source, len(stacks[source])),
            (destination, len(stacks[destination]) + 1),
        ))
        stacks[source].pop()
        stacks[destination].append(container)
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