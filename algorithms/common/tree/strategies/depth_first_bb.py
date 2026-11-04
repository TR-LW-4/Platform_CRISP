"""Depth-first branch-and-bound with interchangeable CRP components."""

from __future__ import annotations

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


class DepthFirstBB(SearchEngine):
    """Depth-first B&B ordered by child lower bounds."""

    def __init__(
        self,
        state_space,
        branching,
        lower_bound,
        upper_bound=None,
    ) -> None:
        self.state_space = state_space
        self.branching = branching
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

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

        self._dfs(problem, root, context, search)
        optimal = not search.aborted and search.best_plan is not None
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

    def _dfs(
        self,
        problem,
        node: SearchNode,
        context: SolveContext,
        search: _SearchState,
    ) -> None:
        search.nodes += 1
        if context.should_stop():
            search.aborted = True
            return

        if self.state_space.is_terminal(node.state):
            if node.g < search.best_cost:
                search.best_cost = node.g
                search.best_plan = reconstruct_plan(node)
            return

        node_bound = node.g + self.lower_bound(problem, node.state)
        if node_bound >= search.best_cost:
            return

        children = []
        for edge in self.branching.expand(problem, node.state):
            child = SearchNode(
                state=edge.state,
                g=node.g + edge.cost,
                parent=node,
                edge=edge,
            )
            child_bound = (
                child.g
                + self.lower_bound(problem, child.state)
            )
            if child_bound < search.best_cost:
                children.append((child_bound, child))

        children.sort(key=lambda item: item[0])
        for _, child in children:
            if search.aborted:
                return
            self._dfs(problem, child, context, search)

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
