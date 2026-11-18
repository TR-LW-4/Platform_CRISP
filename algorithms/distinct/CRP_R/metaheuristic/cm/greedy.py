"""Clear one bay: last empty stack, otherwise the chosen operator."""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple


class Move:
    def __init__(self, container, source, destination):
        self.container = container
        self.source = source
        self.destination = destination


def destination(
    stacks: List[List[int]],
    max_tiers: int,
    choose_destination: Callable,
) -> Optional[int]:
    """Return the destination stack, or None when the target is already on top."""
    priorities = [priority for stack in stacks for priority in stack]
    if not priorities:
        return None
    target = min(priorities)
    source = next(index for index, stack in enumerate(stacks) if target in stack)
    if stacks[source][-1] == target:
        return None
    last_empty = None
    for index, stack in enumerate(stacks):
        if index != source and not stack and len(stack) < max_tiers:
            last_empty = index
    if last_empty is not None:
        return last_empty
    return choose_destination(stacks, max_tiers)


def greedy(
    stacks_init: List[List[int]],
    n_total: int,
    max_tiers: int,
    choose_destination: Callable,
) -> Tuple[List[Move], int]:
    """Return the move list and the number of relocations."""
    state = [list(stack) for stack in stacks_init]
    moves: List[Move] = []
    cost = 0

    for target in range(1, n_total + 1):
        source = next((index for index, stack in enumerate(state) if target in stack), -1)
        if source < 0:
            continue
        while state[source] and state[source][-1] != target:
            container = state[source][-1]
            landing = destination(state, max_tiers, choose_destination)
            if landing is None:
                raise ValueError(
                    f"no free stack for container {container} above target {target}"
                )
            moves.append(Move(
                container,
                (source, len(state[source])),
                (landing, len(state[landing]) + 1),
            ))
            state[source].pop()
            state[landing].append(container)
            cost += 1
        if state[source] and state[source][-1] == target:
            moves.append(Move(target, (source, len(state[source])), None))
            state[source].pop()
    return moves, cost