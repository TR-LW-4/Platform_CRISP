"""Clear one bay: last empty stack, otherwise Min-Max."""

from __future__ import annotations

from typing import List, Optional, Tuple

from algorithms.fixed.CRP_R.operator.min_max import choose_destination

Move = Tuple[int, int, Optional[int]]


def destination(stacks: List[List[int]], max_tiers: int) -> Optional[int]:
    """Return the destination stack, or None when the target is already on top."""
    priorities = [priority for stack in stacks for priority in stack]
    if not priorities:
        return None
    target = min(priorities)
    src = next(index for index, stack in enumerate(stacks) if target in stack)
    if stacks[src][-1] == target:
        return None
    last_empty = None
    for index, stack in enumerate(stacks):
        if index != src and not stack and len(stack) < max_tiers:
            last_empty = index
    if last_empty is not None:
        return last_empty
    return choose_destination(stacks, max_tiers)


def greedy(
    stacks_init: List[List[int]],
    n_total: int,
    max_tiers: int,
) -> Tuple[List[Move], int]:
    """Return the move list and the number of relocations."""
    stacks = [list(stack) for stack in stacks_init]
    moves: List[Move] = []
    cost = 0

    for target in range(1, n_total + 1):
        src = next((index for index, stack in enumerate(stacks) if target in stack), -1)
        if src < 0:
            continue
        while stacks[src] and stacks[src][-1] != target:
            container = stacks[src][-1]
            dst = destination(stacks, max_tiers)
            if dst is None:
                raise ValueError(
                    f"no free stack for container {container} above target {target}"
                )
            moves.append((container, src, dst))
            stacks[src].pop()
            stacks[dst].append(container)
            cost += 1
        if stacks[src] and stacks[src][-1] == target:
            moves.append((target, src, None))
            stacks[src].pop()
    return moves, cost
