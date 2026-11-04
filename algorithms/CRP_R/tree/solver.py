"""Public contract for one-instance solvers."""

from __future__ import annotations

import time
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .base_algorithm import BaseAlgorithm
from .plan import RelocationPlan


@dataclass
class SolverResult:
    """
    Solution returned by a solver.

    status:
        optimal, feasible, time_limit, stopped,
        infeasible, no_solution, invalid_solution, or error.

    details:
        Optional values specific to the algorithm.
    """

    plan: Optional[RelocationPlan]
    objective: Optional[float]
    status: str = "feasible"
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SolveContext:
    """Time and stop control shared with a solver."""

    stop_event: Any
    time_limit_s: Optional[float] = None
    _started_at: float = field(
        default_factory=time.perf_counter,
        init=False,
        repr=False,
    )

    @property
    def elapsed_time(self) -> float:
        return time.perf_counter() - self._started_at

    def remaining_time(self) -> Optional[float]:
        if self.time_limit_s is None:
            return None
        return max(0.0, self.time_limit_s - self.elapsed_time)

    def should_stop(self) -> bool:
        if self.stop_event is not None and self.stop_event.is_set():
            return True
        remaining = self.remaining_time()
        return remaining is not None and remaining <= 0.0


class SolverAlgorithm(BaseAlgorithm):
    """Base class for algorithms that solve one problem instance."""

    @abstractmethod
    def solve(
        self,
        problem: Any,
        context: SolveContext,
    ) -> SolverResult:
        """Return the best result found."""

    def get_best_solution(self) -> Optional[RelocationPlan]:
        return self._best_solution

    def train(
        self,
        problem_factory,
        result_queue,
        stop_event,
    ) -> None:
        from ._solver_runtime import run_solver

        run_solver(
            self,
            problem_factory,
            result_queue,
            stop_event,
        )
