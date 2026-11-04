"""State links used by tree-search strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from core.plan import Movement


@dataclass(frozen=True)
class SearchEdge:
    """One transition in a search tree."""

    state: Any
    moves: Tuple[Movement, ...]
    cost: float


@dataclass
class SearchNode:
    """One node in a search tree."""

    state: Any
    g: float
    parent: Optional["SearchNode"] = None
    edge: Optional[SearchEdge] = None


@dataclass(frozen=True)
class RolloutResult:
    """Completion generated from one state."""

    moves: Tuple[Movement, ...]
    cost: float
    complete: bool
