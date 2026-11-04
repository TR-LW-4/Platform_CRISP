"""Lower bounds for restricted CRP searches."""

from __future__ import annotations

from typing import Optional

from .state import RestrictedCRPSpace, State


class LB1:
    """Count containers that must be relocated."""

    name = "LB1"

    def __init__(self, space: RestrictedCRPSpace) -> None:
        self.space = space

    def __call__(self, problem, state: State) -> int:
        del problem
        total = 0

        for stack in state:
            minimum_below: Optional[int] = None
            for container_id in stack:
                priority = self.space.priority(container_id)
                if (
                    minimum_below is not None
                    and priority > minimum_below
                ):
                    total += 1
                if minimum_below is None:
                    minimum_below = priority
                else:
                    minimum_below = min(minimum_below, priority)
        return total


class LB2(LB1):
    """LB1 plus unavoidable repeated blocker moves."""

    name = "LB2"

    def __call__(self, problem, state: State) -> int:
        total = super().__call__(problem, state)
        target_id = self.space.target_id(state)
        if target_id is None:
            return total

        location = self.space.locate(state, target_id)
        if location is None:
            return total

        source, target_tier = location
        for container_id in state[source][target_tier + 1:]:
            if not _has_safe_destination(
                self.space,
                state,
                source,
                container_id,
            ):
                total += 1
        return total


class LB3(LB1):
    """LB1 plus future unavoidable repeated moves."""

    name = "LB3"

    def __call__(self, problem, state: State) -> int:
        total = super().__call__(problem, state)

        for source, stack in enumerate(state):
            if not stack:
                continue

            local_target = min(
                stack,
                key=self.space.priority,
            )
            target_priority = self.space.priority(local_target)
            target_tier = stack.index(local_target)
            blockers = stack[target_tier + 1:]
            if not blockers:
                continue

            reduced = []
            for current_stack in state:
                cut = len(current_stack)
                for index, container_id in enumerate(current_stack):
                    if self.space.priority(container_id) < target_priority:
                        cut = index
                        break
                reduced.append(current_stack[:cut])
            reduced_state = tuple(reduced)

            for container_id in blockers:
                if not _has_safe_destination(
                    self.space,
                    reduced_state,
                    source,
                    container_id,
                ):
                    total += 1
        return total


def _has_safe_destination(
    space: RestrictedCRPSpace,
    state: State,
    source: int,
    container_id: int,
) -> bool:
    priority = space.priority(container_id)

    for destination in space.eligible_destinations(state, source):
        stack = state[destination]
        if not stack:
            return True
        if min(space.priority(current_id) for current_id in stack) > priority:
            return True
    return False
