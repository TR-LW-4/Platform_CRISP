"""Public contracts for tree-search algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.solver import SolveContext, SolverAlgorithm, SolverResult


class SearchEngine(ABC):
    """Search strategy built by a tree solver."""

    @abstractmethod
    def solve(
        self,
        problem: Any,
        context: SolveContext,
    ) -> SolverResult:
        """Search one problem instance."""


class TreeSolver(SolverAlgorithm):
    """Base class for tree-search algorithms."""

    @abstractmethod
    def build_search(self) -> SearchEngine:
        """Return the configured search strategy."""

    def solve(
        self,
        problem: Any,
        context: SolveContext,
    ) -> SolverResult:
        return self.build_search().solve(problem, context)
