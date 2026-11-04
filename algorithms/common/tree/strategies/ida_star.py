"""Component-based iterative deepening A*."""

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


class IDAStar(SearchEngine):
    """IDA* configured with interchangeable components."""

    def __init__(
        self,
        state_space,
        branching,
        lower_bound,
        upper_bound,
        frontier_probe=None,
    ) -> None:
        self.state_space = state_space
        self.branching = branching
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.frontier_probe = frontier_probe

    def solve(
        self,
        problem: Any,
        context: SolveContext,
    ) -> SolverResult:
        if context.should_stop():
            status = (
                "stopped"
                if context.stop_event is not None
                and context.stop_event.is_set()
                else "time_limit"
            )
            return SolverResult(None, None, status)

        root_edge = self.state_space.initial(problem)
        root = SearchNode(
            state=root_edge.state,
            g=0.0,
            edge=root_edge,
        )
        search = _SearchState()

        if self.state_space.is_terminal(root.state):
            return SolverResult(
                plan=reconstruct_plan(root),
                objective=0.0,
                status="optimal",
                details={
                    "nodes_explored": 1.0,
                    "root_lower_bound": 0.0,
                    "optimal_proven": 1.0,
                },
            )

        initial = self.upper_bound(problem, root.state)
        self._update_from_rollout(root, initial, search)

        root_bound = float(self.lower_bound(problem, root.state))
        threshold = root_bound
        optimal = False

        while not search.aborted:
            next_threshold = self._dfs(
                problem,
                root,
                threshold,
                context,
                search,
            )
            if search.aborted:
                break
            if search.best_cost <= threshold:
                optimal = True
                break
            if next_threshold == float("inf"):
                optimal = search.best_plan is not None
                break
            threshold = max(threshold + 1.0, next_threshold)

        status = self._status(context, search, optimal)
        objective = (
            search.best_cost
            if search.best_plan is not None
            else None
        )
        return SolverResult(
            plan=search.best_plan,
            objective=objective,
            status=status,
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
        threshold: float,
        context: SolveContext,
        search: _SearchState,
    ) -> float:
        search.nodes += 1
        if context.should_stop():
            search.aborted = True
            return float("inf")

        if self.state_space.is_terminal(node.state):
            if node.g < search.best_cost:
                search.best_cost = node.g
                search.best_plan = reconstruct_plan(node)
            return float("inf")

        node_bound = node.g + self.lower_bound(problem, node.state)
        if node_bound > threshold:
            if (
                node_bound < search.best_cost
                and self.frontier_probe is not None
            ):
                rollout = self.frontier_probe(problem, node.state)
                self._update_from_rollout(node, rollout, search)
            return node_bound

        if node_bound >= search.best_cost:
            return float("inf")

        next_threshold = float("inf")
        for edge in self.branching.expand(problem, node.state):
            if search.aborted:
                break
            child = SearchNode(
                state=edge.state,
                g=node.g + edge.cost,
                parent=node,
                edge=edge,
            )
            child_threshold = self._dfs(
                problem,
                child,
                threshold,
                context,
                search,
            )
            next_threshold = min(next_threshold, child_threshold)
        return next_threshold

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
