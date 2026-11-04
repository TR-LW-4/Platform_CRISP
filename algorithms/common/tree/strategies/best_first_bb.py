"""Best-first branch-and-bound with a paper-specific child policy."""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass
from typing import Any, Optional

from core.plan import RelocationPlan
from core.solver import SolveContext, SolverResult

from ..model import RolloutResult, SearchNode
from ..path import reconstruct_plan
from ..solver import SearchEngine


@dataclass
class _SearchState:
    best_cost: float = float("inf")
    best_plan: Optional[RelocationPlan] = None
    nodes: int = 0
    aborted: bool = False


class BestFirstBB(SearchEngine):
    """Explore nodes in ascending ``g + lower_bound`` order."""

    def __init__(
        self,
        state_space,
        branching,
        lower_bound,
        upper_bound=None,
        child_policy=None,
        complete_search: bool = True,
    ) -> None:
        self.state_space = state_space
        self.branching = branching
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.child_policy = child_policy
        self.complete_search = complete_search

    def solve(
        self,
        problem: Any,
        context: SolveContext,
    ) -> SolverResult:
        root_edge = self.state_space.initial(problem)
        root = SearchNode(
            state=root_edge.state,
            g=0.0,
            edge=root_edge,
        )
        root_bound = float(self.lower_bound(problem, root.state))
        search = _SearchState()

        if self.upper_bound is not None:
            rollout = self.upper_bound(problem, root.state)
            self._update_from_rollout(root, rollout, search)

        counter = itertools.count()
        queue = [(root_bound, next(counter), root)]
        finished = False

        while queue:
            if context.should_stop():
                search.aborted = True
                break

            node_bound, _, node = heapq.heappop(queue)
            if node_bound >= search.best_cost:
                finished = True
                break

            search.nodes += 1
            if self.state_space.is_terminal(node.state):
                self._update_from_node(node, search)
                continue

            edges = list(
                self.branching.expand(problem, node.state)
            )
            if self.child_policy is not None:
                edges = list(
                    self.child_policy(problem, node, edges)
                )

            for edge in edges:
                child = SearchNode(
                    state=edge.state,
                    g=node.g + edge.cost,
                    parent=node,
                    edge=edge,
                )
                if self.state_space.is_terminal(child.state):
                    self._update_from_node(child, search)
                    continue

                child_bound = (
                    child.g
                    + self.lower_bound(problem, child.state)
                )
                if child_bound < search.best_cost:
                    heapq.heappush(
                        queue,
                        (child_bound, next(counter), child),
                    )
        else:
            finished = True

        optimal = (
            self.complete_search
            and finished
            and not search.aborted
            and search.best_plan is not None
        )
        return SolverResult(
            plan=search.best_plan,
            objective=(
                search.best_cost
                if search.best_plan is not None
                else None
            ),
            status=self._status(context, search, optimal),
            details={
                "nodes_explored": float(search.nodes),
                "root_lower_bound": root_bound,
                "optimal_proven": float(optimal),
            },
        )

    @staticmethod
    def _update_from_node(
        node: SearchNode,
        search: _SearchState,
    ) -> None:
        if node.g < search.best_cost:
            search.best_cost = node.g
            search.best_plan = reconstruct_plan(node)

    @staticmethod
    def _update_from_rollout(
        node: SearchNode,
        rollout: RolloutResult,
        search: _SearchState,
    ) -> None:
        if not rollout.complete:
            return

        total = node.g + rollout.cost
        if total >= search.best_cost:
            return

        prefix = reconstruct_plan(node)
        search.best_cost = total
        search.best_plan = RelocationPlan(
            prefix.movements + list(rollout.moves)
        )

    @staticmethod
    def _status(
        context: SolveContext,
        search: _SearchState,
        optimal: bool,
    ) -> str:
        if optimal:
            return "optimal"
        if (
            context.stop_event is not None
            and context.stop_event.is_set()
        ):
            return "stopped"
        if search.aborted:
            return "time_limit"
        if search.best_plan is not None:
            return "feasible"
        return "no_solution"
