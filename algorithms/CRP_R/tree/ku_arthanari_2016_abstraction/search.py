"""SearchEngine adapter for Ku and Arthanari's abstraction method."""

from __future__ import annotations

import time
from typing import Any

from algorithms.common.tree import SearchEngine
from algorithms.common.tree.components import RestrictedCRPSpace
from core.plan import RelocationPlan
from core.solver import SolveContext, SolverResult

from .abstraction_core import build_pdb, solve


class KuAbstractionSearch(SearchEngine):
    """Forward DFBnB joined with an exact backward PDB."""

    def __init__(
        self,
        state_space: RestrictedCRPSpace,
        pdb_depth: int,
        cache_depth: int,
        max_cache_size: int,
        pdb_build_time_limit_s: float,
    ) -> None:
        self.state_space = state_space
        self.pdb_depth = max(0, int(pdb_depth))
        self.cache_depth = max(0, int(cache_depth))
        self.max_cache_size = max(1, int(max_cache_size))
        self.pdb_build_time_limit_s = max(
            0.0,
            float(pdb_build_time_limit_s),
        )

    def solve(
        self,
        problem: Any,
        context: SolveContext,
    ) -> SolverResult:
        root = self.state_space.initial(problem)
        if self.state_space.is_terminal(root.state):
            return SolverResult(
                plan=RelocationPlan(list(root.moves)),
                objective=0.0,
                status="optimal",
                details={
                    "nodes_explored": 1.0,
                    "root_lower_bound": 0.0,
                    "optimal_proven": 1.0,
                },
            )

        n_total = len(self.state_space.priorities)
        stacks = [
            [
                self.state_space.priority(container_id)
                for container_id in stack
            ]
            for stack in root.state
        ]

        build_limit = self.pdb_build_time_limit_s
        remaining = context.remaining_time()
        if remaining is not None:
            build_limit = min(build_limit, remaining)

        build_started = time.perf_counter()
        levels = build_pdb(
            depth=self.pdb_depth,
            n_cols=len(stacks),
            m_tiers=self.state_space.max_tiers,
            time_limit_s=build_limit,
            stop_event=context.stop_event,
        )
        build_elapsed = time.perf_counter() - build_started

        if context.should_stop():
            return SolverResult(
                plan=None,
                objective=None,
                status=self._stop_status(context),
                details={
                    "nodes_explored": 0.0,
                    "pdb_build_time_s": build_elapsed,
                    "pdb_depth_built": float(max(levels)),
                    "optimal_proven": 0.0,
                },
            )

        remaining = context.remaining_time()
        search_limit = (
            float("inf")
            if remaining is None
            else remaining
        )
        result = solve(
            stacks,
            n_total,
            self.state_space.max_tiers,
            pdb_levels=levels,
            pdb_depth=max(levels),
            cache_depth=self.cache_depth,
            max_cache_size=self.max_cache_size,
            time_limit_s=search_limit,
            stop_event=context.stop_event,
        )

        plan = self._replay(root, result["path"])
        complete = (
            plan is not None
            and plan.num_retrievals() == n_total
        )
        optimal = bool(result["optimal"] and complete)

        if optimal:
            status = "optimal"
        elif complete:
            status = "feasible"
        elif result["aborted"]:
            status = self._stop_status(context)
        else:
            status = "no_solution"

        return SolverResult(
            plan=plan if complete else None,
            objective=(
                float(result["best"])
                if complete
                else None
            ),
            status=status,
            details={
                "nodes_explored": float(result["nodes"]),
                "root_lower_bound": float(result["root_lb"]),
                "cache_hits": float(result["cache_hits"]),
                "pdb_hits": float(result["pdb_hits"]),
                "pdb_depth_built": float(max(levels)),
                "pdb_build_time_s": build_elapsed,
                "optimal_proven": float(optimal),
            },
        )

    def _replay(self, root, path):
        current = root.state
        moves = list(root.moves)

        for source, destination in path:
            edge = self.state_space.relocate(
                current,
                source,
                destination,
            )
            current = edge.state
            moves.extend(edge.moves)

        if not self.state_space.is_terminal(current):
            return None
        return RelocationPlan(moves)

    @staticmethod
    def _stop_status(context: SolveContext) -> str:
        if (
            context.stop_event is not None
            and context.stop_event.is_set()
        ):
            return "stopped"
        return "time_limit"
