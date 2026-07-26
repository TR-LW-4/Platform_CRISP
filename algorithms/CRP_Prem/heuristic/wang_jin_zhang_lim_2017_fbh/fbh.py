"""
Algorithm 1: the main Feasibility-Based Heuristic (FBH) loop.

    (L, f) <- (L0, U * 1)                       # pre-fix unreachable tiers
    if (L, f) is a dead end: fail
    repeat N - S*U times:
        select task (c -> s*) minimizing the six-tuple over valid tasks T
        if T is empty: fail
        accomplish the task via STAP
        f(s*) <- f(s*) + 1
    return the move sequence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .feasibility import (
    compute_all_stable_heights,
    compute_surplus,
    is_dead_end_state,
)
from .stap import InfeasibleInstance, accomplish_task
from .state import (
    Move,
    State,
    initialize_fixed_from_unreachable,
    orderly_height,
    total_containers,
)
from .task_selection import select_task


@dataclass
class FBHResult:
    moves: List[Move] = field(default_factory=list)
    solved: bool = False
    reason: str = ""
    iterations: int = 0


def is_solved(state: State) -> bool:
    return all(orderly_height(arr) == len(arr) for arr in state.stacks.values())


def fbh_solve(state: State, p1: int = 0, p2: int = 1, initialize: bool = True) -> FBHResult:
    """
    Run FBH to completion (or until it reports infeasibility) on ``state``,
    mutating it in place. ``p1``/``p2`` are the tier-protection thresholds
    of §5.2.2 (a stack with ``f(s) <= p1`` and affected demand ``>= p2`` is
    tier-protected).
    """
    if initialize:
        initialize_fixed_from_unreachable(state)

    if is_dead_end_state(state):
        return FBHResult(solved=False, reason="initial state is a dead end")

    n_total = total_containers(state)
    n_prefixed = sum(state.fixed_height.values())
    iterations = max(n_total - n_prefixed, 0)

    moves: List[Move] = []
    try:
        for _ in range(iterations):
            surplus = compute_surplus(state)
            stable_heights = compute_all_stable_heights(state)
            task = select_task(state, p1, p2, stable_heights=stable_heights, surplus=surplus)
            if task is None:
                return FBHResult(moves=moves, solved=False, reason="no valid task in T", iterations=len(moves))
            c_pos, s_star = task
            accomplish_task(state, c_pos, s_star, moves)
            state.fixed_height[s_star] += 1
    except InfeasibleInstance as exc:
        return FBHResult(moves=moves, solved=False, reason=str(exc), iterations=len(moves))

    solved = is_solved(state)
    reason = "" if solved else "loop completed but final layout is not fully orderly"
    return FBHResult(moves=moves, solved=solved, reason=reason, iterations=len(moves))
