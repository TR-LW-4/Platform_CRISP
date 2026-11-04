"""PR rollout heuristics for restricted CRP searches."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from core.plan import Movement

from ..model import RolloutResult
from .state import RestrictedCRPSpace, State

PR1 = "PR1"
PR2 = "PR2"
PR3 = "PR3"
PR4 = "PR4"


class PRRollout:
    """Complete a state with one PR destination rule."""

    def __init__(
        self,
        space: RestrictedCRPSpace,
        mode: str = PR4,
    ) -> None:
        self.space = space
        self.mode = mode if mode in _DESTINATIONS else PR4

    def __call__(self, problem, state: State) -> RolloutResult:
        del problem
        current = state
        moves: List[Movement] = []
        cost = 0.0
        guard = (
            sum(len(stack) for stack in state)
            * max(self.space.max_tiers, 1)
            * max(len(state), 1)
            + 16
        )

        while not self.space.is_terminal(current) and cost <= guard:
            target_id = self.space.target_id(current)
            if target_id is None:
                break

            location = self.space.locate(current, target_id)
            if location is None:
                break

            source, _ = location
            blocker_id = current[source][-1]
            destination = _DESTINATIONS[self.mode](
                self.space,
                current,
                source,
                blocker_id,
            )
            if destination is None:
                break

            edge = self.space.relocate(
                current,
                source,
                destination,
            )
            current = edge.state
            moves.extend(edge.moves)
            cost += edge.cost

        return RolloutResult(
            moves=tuple(moves),
            cost=cost,
            complete=self.space.is_terminal(current),
        )


class PRPlus:
    """Return the best complete PR1-PR4 rollout."""

    def __init__(self, space: RestrictedCRPSpace) -> None:
        self.rollouts = tuple(
            PRRollout(space, mode)
            for mode in (PR1, PR2, PR3, PR4)
        )

    def __call__(self, problem, state: State) -> RolloutResult:
        results = [
            rollout(problem, state)
            for rollout in self.rollouts
        ]
        complete = [
            result
            for result in results
            if result.complete
        ]
        if complete:
            return min(complete, key=lambda result: result.cost)
        return RolloutResult(moves=(), cost=float("inf"), complete=False)


def _minimum_priority(
    space: RestrictedCRPSpace,
    stack: Tuple[int, ...],
) -> float:
    if not stack:
        return float("inf")
    return min(space.priority(container_id) for container_id in stack)


def _eligible(
    space: RestrictedCRPSpace,
    state: State,
    source: int,
) -> List[int]:
    return space.eligible_destinations(state, source)


def _pr1(
    space: RestrictedCRPSpace,
    state: State,
    source: int,
    blocker_id: int,
) -> Optional[int]:
    del blocker_id
    destinations = _eligible(space, state, source)
    if not destinations:
        return None
    return min(
        destinations,
        key=lambda index: (len(state[index]), index),
    )


def _pr2(
    space: RestrictedCRPSpace,
    state: State,
    source: int,
    blocker_id: int,
) -> Optional[int]:
    del blocker_id
    destinations = _eligible(space, state, source)
    if not destinations:
        return None
    return max(
        destinations,
        key=lambda index: (
            _minimum_priority(space, state[index]),
            -index,
        ),
    )


def _pr3(
    space: RestrictedCRPSpace,
    state: State,
    source: int,
    blocker_id: int,
) -> Optional[int]:
    destinations = _eligible(space, state, source)
    if not destinations:
        return None

    blocker_priority = space.priority(blocker_id)
    safe = [
        index
        for index in destinations
        if _minimum_priority(space, state[index]) > blocker_priority
    ]
    if safe:
        return min(
            safe,
            key=lambda index: _minimum_priority(space, state[index]),
        )
    return max(
        destinations,
        key=lambda index: _minimum_priority(space, state[index]),
    )


def _pr4(
    space: RestrictedCRPSpace,
    state: State,
    source: int,
    blocker_id: int,
) -> Optional[int]:
    destinations = _eligible(space, state, source)
    if not destinations:
        return None

    blocker_priority = space.priority(blocker_id)
    safe = [
        index
        for index in destinations
        if _minimum_priority(space, state[index]) > blocker_priority
    ]
    if safe:
        return min(
            safe,
            key=lambda index: _minimum_priority(space, state[index]),
        )

    ranked = sorted(
        destinations,
        key=lambda index: -_minimum_priority(space, state[index]),
    )
    best = ranked[0]
    source_priorities = sorted(
        space.priority(container_id)
        for container_id in state[source]
    )
    second_smallest = (
        len(source_priorities) >= 2
        and blocker_priority == source_priorities[1]
    )
    one_slot = _free_tiers(space, state, best) == 1
    if not second_smallest and one_slot and len(ranked) > 1:
        return ranked[1]
    return best


def _free_tiers(
    space: RestrictedCRPSpace,
    state: State,
    stack_index: int,
) -> int:
    """Return free tiers in one stack."""
    return space.max_tiers - len(state[stack_index])


_DESTINATIONS: Dict[
    str,
    Callable[
        [RestrictedCRPSpace, State, int, int],
        Optional[int],
    ],
] = {
    PR1: _pr1,
    PR2: _pr2,
    PR3: _pr3,
    PR4: _pr4,
}
