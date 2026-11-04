"""Level-wise beam search with paper-specific state evaluation."""

from __future__ import annotations

from typing import Any, List, Tuple

from core.solver import SolveContext, SolverResult

from ..model import SearchNode
from ..path import reconstruct_plan
from ..solver import SearchEngine


class BeamSearch(SearchEngine):
    """Retain the best-scoring nodes at each relocation level."""

    def __init__(
        self,
        state_space,
        branching,
        evaluator,
        width: int,
        diverse_parents: bool = True,
    ) -> None:
        self.state_space = state_space
        self.branching = branching
        self.evaluator = evaluator
        self.width = max(1, int(width))
        self.diverse_parents = diverse_parents

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
        if self.state_space.is_terminal(root.state):
            return SolverResult(
                plan=reconstruct_plan(root),
                objective=0.0,
                status="optimal",
                details={
                    "nodes_explored": 1.0,
                    "optimal_proven": 1.0,
                },
            )

        beam = [root]
        nodes = 0

        while beam:
            if context.should_stop():
                return SolverResult(
                    plan=None,
                    objective=None,
                    status=self._stop_status(context),
                    details={
                        "nodes_explored": float(nodes),
                        "optimal_proven": 0.0,
                    },
                )

            children: List[Tuple[SearchNode, float, int]] = []
            for parent_index, parent in enumerate(beam):
                nodes += 1
                for edge in self.branching.expand(
                    problem,
                    parent.state,
                ):
                    child = SearchNode(
                        state=edge.state,
                        g=parent.g + edge.cost,
                        parent=parent,
                        edge=edge,
                    )
                    if self.state_space.is_terminal(child.state):
                        return SolverResult(
                            plan=reconstruct_plan(child),
                            objective=child.g,
                            status="feasible",
                            details={
                                "nodes_explored": float(nodes),
                                "optimal_proven": 0.0,
                            },
                        )
                    score = float(
                        self.evaluator(problem, child.state)
                    )
                    children.append(
                        (child, score, parent_index)
                    )

            if not children:
                break
            beam = self._select(children)

        return SolverResult(
            plan=None,
            objective=None,
            status="no_solution",
            details={
                "nodes_explored": float(nodes),
                "optimal_proven": 0.0,
            },
        )

    def _select(
        self,
        children: List[Tuple[SearchNode, float, int]],
    ) -> List[SearchNode]:
        children.sort(key=lambda item: item[1])
        if len(children) <= self.width:
            return [item[0] for item in children]

        threshold = children[self.width - 1][1]
        selected = [
            item
            for item in children
            if item[1] < threshold
        ]
        tied = [
            item
            for item in children
            if item[1] == threshold
        ]
        remaining = self.width - len(selected)

        if self.diverse_parents:
            seen = {item[2] for item in selected}
            fresh = [item for item in tied if item[2] not in seen]
            repeated = [item for item in tied if item[2] in seen]
            tied = fresh + repeated

        selected.extend(tied[:remaining])
        return [item[0] for item in selected]

    @staticmethod
    def _stop_status(context: SolveContext) -> str:
        if (
            context.stop_event is not None
            and context.stop_event.is_set()
        ):
            return "stopped"
        return "time_limit"
