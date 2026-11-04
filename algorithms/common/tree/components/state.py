"""Lightweight CRP-R state operations."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from core.plan import Movement

from ..model import SearchEdge

State = Tuple[Tuple[int, ...], ...]
Position = Tuple[int, int]


class RestrictedCRPSpace:
    """State operations shared by restricted CRP searches."""

    def __init__(self) -> None:
        self.positions: Tuple[Position, ...] = ()
        self.priorities: Dict[int, int] = {}
        self.max_tiers = 0

    def initial(self, problem) -> SearchEdge:
        """Return the normalized initial state."""
        problem.reset(options={"skip_auto_retrieve": True})

        stacks = sorted(
            problem.yard.stacks.values(),
            key=lambda stack: (stack.bay, stack.row),
        )
        self.positions = tuple(
            (stack.bay, stack.row)
            for stack in stacks
        )
        self.priorities = {
            container.id: container.priority
            for container in problem.containers
        }
        self.max_tiers = int(problem.config.max_tiers)

        state = tuple(
            tuple(container.id for container in stack.containers)
            for stack in stacks
        )
        state, moves = self.retrieve_ready(state)
        return SearchEdge(state=state, moves=moves, cost=0.0)

    def priority(self, container_id: int) -> int:
        return self.priorities[container_id]

    def target_id(self, state: State) -> Optional[int]:
        remaining = [
            container_id
            for stack in state
            for container_id in stack
        ]
        if not remaining:
            return None
        return min(
            remaining,
            key=lambda container_id: (
                self.priority(container_id),
                container_id,
            ),
        )

    @staticmethod
    def locate(
        state: State,
        container_id: int,
    ) -> Optional[Tuple[int, int]]:
        for stack_index, stack in enumerate(state):
            for tier_index, current_id in enumerate(stack):
                if current_id == container_id:
                    return stack_index, tier_index
        return None

    @staticmethod
    def is_terminal(state: State) -> bool:
        return all(not stack for stack in state)

    def retrieve_ready(
        self,
        state: State,
    ) -> Tuple[State, Tuple[Movement, ...]]:
        """Retrieve consecutive accessible targets."""
        stacks = [list(stack) for stack in state]
        moves: List[Movement] = []

        while True:
            current = tuple(tuple(stack) for stack in stacks)
            target_id = self.target_id(current)
            if target_id is None:
                return current, tuple(moves)

            location = self.locate(current, target_id)
            if location is None:
                return current, tuple(moves)

            stack_index, tier_index = location
            if tier_index != len(stacks[stack_index]) - 1:
                return current, tuple(moves)

            moves.append(
                Movement(
                    container_id=target_id,
                    from_pos=self.positions[stack_index],
                    to_pos=None,
                    from_tier=len(stacks[stack_index]),
                    to_tier=None,
                )
            )
            stacks[stack_index].pop()

    def eligible_destinations(
        self,
        state: State,
        source: int,
    ) -> List[int]:
        return [
            index
            for index, stack in enumerate(state)
            if index != source and len(stack) < self.max_tiers
        ]

    def relocate(
        self,
        state: State,
        source: int,
        destination: int,
    ) -> SearchEdge:
        stacks = [list(stack) for stack in state]
        container_id = stacks[source].pop()
        from_tier = len(stacks[source]) + 1
        to_tier = len(stacks[destination]) + 1
        stacks[destination].append(container_id)

        relocation = Movement(
            container_id=container_id,
            from_pos=self.positions[source],
            to_pos=self.positions[destination],
            from_tier=from_tier,
            to_tier=to_tier,
        )
        child = tuple(tuple(stack) for stack in stacks)
        child, retrievals = self.retrieve_ready(child)
        return SearchEdge(
            state=child,
            moves=(relocation,) + retrievals,
            cost=1.0,
        )
