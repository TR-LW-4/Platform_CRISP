"""Run one unrestricted relocation rule until the bay is empty."""

from __future__ import annotations

import importlib
from typing import Callable, List, Optional, Sequence, Tuple

from algorithms.ALGORITHM import Algorithm
from algorithms.fixed.CRP_U.move import Move

Operator = Callable[[Sequence[Sequence[int]], int], Optional[Tuple[int, int]]]


def rollout(stacks, max_tiers: int, choose_move: Operator) -> List[Move]:
    state = [list(stack) for stack in stacks]
    moves: List[Move] = []
    while True:
        present = [priority for stack in state for priority in stack]
        if not present:
            return moves
        target = min(present)
        home = next(index for index, stack in enumerate(state) if target in stack)
        if state[home][-1] == target:
            moves.append(Move(target, home, None))
            state[home].pop()
            continue
        chosen = choose_move(state, max_tiers)
        if chosen is None:
            raise ValueError(f"operator has no relocation while {target} is buried")
        source, destination = chosen
        if source == destination or not state[source] or len(state[destination]) >= max_tiers:
            raise ValueError(f"operator returned an illegal relocation {chosen}")
        container = state[source].pop()
        state[destination].append(container)
        moves.append(Move(container, source, destination))


class Rollout(Algorithm):

    def main(self):
        name = self.parameter("operator", "dsk", str)
        module = importlib.import_module(f"algorithms.fixed.CRP_U.operator.{name}")
        return rollout(self.problem.stacks, self.problem.max_tiers, module.choose_move)
