"""Path reconstruction for tree-search nodes."""

from __future__ import annotations

from typing import List

from core.plan import Movement, RelocationPlan

from .model import SearchNode


def reconstruct_plan(node: SearchNode) -> RelocationPlan:
    """Return all root-to-node movements."""
    chunks: List[List[Movement]] = []
    current = node

    while current is not None:
        if current.edge is not None:
            chunks.append(list(current.edge.moves))
        current = current.parent

    movements: List[Movement] = []
    for chunk in reversed(chunks):
        movements.extend(chunk)
    return RelocationPlan(movements)
