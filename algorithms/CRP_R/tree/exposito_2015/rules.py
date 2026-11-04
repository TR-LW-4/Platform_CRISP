"""Paper-specific ranking and greedy completion for Expósito et al."""

from __future__ import annotations

from typing import Iterable, List, Optional

from algorithms.common.tree import RolloutResult, SearchEdge, SearchNode
from algorithms.common.tree.components import RestrictedCRPSpace
from core.plan import Movement


class ExpositoPolicy:
    """Rank block assignments by Equation 19 and apply alpha."""

    def __init__(
        self,
        space: RestrictedCRPSpace,
        lower_bound,
        alpha: Optional[int],
    ) -> None:
        self.space = space
        self.lower_bound = lower_bound
        self.alpha = alpha

    def __call__(
        self,
        problem,
        node: SearchNode,
        edges: Iterable[SearchEdge],
    ) -> List[SearchEdge]:
        scored = [
            (
                self.score(problem, node.state, edge),
                edge,
            )
            for edge in edges
        ]
        scored.sort(key=lambda item: item[0])
        if self.alpha is not None:
            scored = scored[:self.alpha]
        return [edge for _, edge in scored]

    def score(self, problem, state, edge: SearchEdge) -> float:
        base = float(self.lower_bound(problem, state))
        position_index = {
            position: index
            for index, position in enumerate(self.space.positions)
        }
        n_total = len(self.space.priorities)
        stack_values = [
            (
                min(
                    self.space.priority(container_id)
                    for container_id in stack
                )
                if stack
                else n_total + 1
            )
            for stack in state
        ]

        total = 0
        for move in edge.moves:
            if move.is_retrieval or move.to_pos is None:
                continue
            destination = position_index[move.to_pos]
            total += stack_values[destination]
        return base + 1.0 / (1.0 + total)


class ExpositoGreedy:
    """Complete a state by repeatedly taking the best assignment."""

    def __init__(
        self,
        space: RestrictedCRPSpace,
        branching,
        policy: ExpositoPolicy,
    ) -> None:
        self.space = space
        self.branching = branching
        self.policy = policy

    def __call__(self, problem, state) -> RolloutResult:
        current = state
        moves: List[Movement] = []
        cost = 0.0

        while not self.space.is_terminal(current):
            node = SearchNode(state=current, g=cost)
            edges = list(self.branching.expand(problem, current))
            ranked = self.policy(problem, node, edges)
            if not ranked:
                return RolloutResult(
                    moves=tuple(moves),
                    cost=float("inf"),
                    complete=False,
                )

            edge = ranked[0]
            current = edge.state
            moves.extend(edge.moves)
            cost += edge.cost

        return RolloutResult(
            moves=tuple(moves),
            cost=cost,
            complete=True,
        )
