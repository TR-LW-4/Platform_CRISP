"""Branching rules shared by CRP tree searches."""

from __future__ import annotations

from typing import Iterable, List

from ..model import SearchEdge
from .state import RestrictedCRPSpace, State


class SingleMove:
    """Move one blocker from the current target stack."""

    def __init__(self, space: RestrictedCRPSpace) -> None:
        self.space = space

    def expand(
        self,
        problem,
        state: State,
    ) -> Iterable[SearchEdge]:
        del problem

        target_id = self.space.target_id(state)
        if target_id is None:
            return

        location = self.space.locate(state, target_id)
        if location is None:
            return

        source, _ = location
        for destination in self.space.eligible_destinations(state, source):
            yield self.space.relocate(
                state,
                source,
                destination,
            )


class BlockAssignment:
    """Assign every blocker above the current target in one branch."""

    def __init__(self, space: RestrictedCRPSpace) -> None:
        self.space = space

    def expand(
        self,
        problem,
        state: State,
    ) -> Iterable[SearchEdge]:
        del problem

        target_id = self.space.target_id(state)
        if target_id is None:
            return

        location = self.space.locate(state, target_id)
        if location is None:
            return

        source, target_tier = location
        blockers = len(state[source]) - target_tier - 1
        if blockers <= 0:
            return

        yield from self._assign(
            state,
            source,
            blockers,
            (),
            0.0,
        )

    def _assign(
        self,
        state: State,
        source: int,
        remaining: int,
        moves: tuple,
        cost: float,
    ) -> Iterable[SearchEdge]:
        if remaining == 0:
            yield SearchEdge(
                state=state,
                moves=moves,
                cost=cost,
            )
            return

        destinations: List[int] = self.space.eligible_destinations(
            state,
            source,
        )
        for destination in destinations:
            edge = self.space.relocate(
                state,
                source,
                destination,
            )
            yield from self._assign(
                edge.state,
                source,
                remaining - 1,
                moves + edge.moves,
                cost + edge.cost,
            )
