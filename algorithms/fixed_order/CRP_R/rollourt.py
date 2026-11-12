"""Finish one bay with a destination operator and return the path and relocation count."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

Move = Tuple[int, int, Optional[int]]
Operator = Callable[[Sequence[Sequence[int]], int], Optional[int]]


def rollout(stacks, max_tiers: int, choose_destination: Operator) -> Tuple[List[Move], int]:
    """Relocate with the operator, retrieve exposed targets, and return the path.

    A move is ``(container, source, destination)``. ``destination is None``
    marks a retrieval. The count includes relocations only.
    """
    state = [list(stack) for stack in stacks]
    moves: List[Move] = []
    cost = 0

    while True:
        present = [priority for stack in state for priority in stack]
        if not present:
            return moves, cost
        target = min(present)
        source = next(index for index, stack in enumerate(state) if target in stack)
        if state[source][-1] == target:
            moves.append((target, source, None))
            state[source].pop()
            continue

        destination = choose_destination(state, max_tiers)
        if destination is None or destination == source or len(state[destination]) >= max_tiers:
            raise ValueError(f"operator has no stack for the container above {target}")
        container = state[source][-1]
        moves.append((container, source, destination))
        state[destination].append(state[source].pop())
        cost += 1