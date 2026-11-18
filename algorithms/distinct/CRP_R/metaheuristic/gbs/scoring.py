"""Completion scores for the greedy beam search."""

from __future__ import annotations

from typing import Sequence

from algorithms.distinct.CRP_R.operator.sdh import choose_destination
from algorithms.distinct.CRP_R.operator.vrh import moves


def sdh_cost(stacks: Sequence[Sequence[int]], n_total: int, max_tiers: int) -> int:
    """Relocation count of walking the stack-difference rule one step at a time."""
    state = [list(stack) for stack in stacks]
    relocations = 0
    for target in range(1, n_total + 1):
        source = next((index for index, stack in enumerate(state) if target in stack), None)
        if source is None:
            continue
        while state[source] and state[source][-1] != target:
            destination = choose_destination(state, max_tiers)
            if destination is None:
                return n_total * max_tiers
            container = state[source].pop()
            state[destination].append(container)
            relocations += 1
        if state[source] and state[source][-1] == target:
            state[source].pop()
    return relocations


def vrh_cost(stacks: Sequence[Sequence[int]], n_total: int, max_tiers: int) -> int:
    """Relocation count of the one-shot VRH completion."""
    try:
        plan = moves(stacks, max_tiers)
    except ValueError:
        return n_total * max_tiers
    return sum(1 for step in plan if step.destination is not None)