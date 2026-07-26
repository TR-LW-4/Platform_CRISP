"""
Deterministic CRP tail solver ported from:

- Astar.m
- Astar_Rec.m

This solver is only used once the full retrieval order is known, exactly
as in the source repository.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .bay_utils import preprocess_bay
from .bounds import blocking_lower_bound, rolling_lower_bound
from .heuristics import run_heuristic_batch


UPPER_BOUND_HEURISTICS = {
    1: "EG",
    2: "EM",
    3: "ERI",
    4: "L",
    5: "Rand",
}


def astar(
    bay: np.ndarray,
    lower_bound_type: int = 1,
    upper_bound_type: int = 2,
    rng: Optional[np.random.RandomState] = None,
) -> float:
    """Port of `Astar.m`."""
    rng = rng if rng is not None else np.random.RandomState()
    reloc = 0
    best_incumbent = reloc + run_heuristic_batch(
        bay, upper_bound_type, 1, rng=rng
    )
    return _astar_rec(
        bay.copy(),
        lower_bound_type=lower_bound_type,
        upper_bound_type=upper_bound_type,
        best_incumbent=best_incumbent,
        reloc=reloc,
        rng=rng,
    )


def _astar_rec(
    bay: np.ndarray,
    lower_bound_type: int,
    upper_bound_type: int,
    best_incumbent: float,
    reloc: int,
    rng: np.random.RandomState,
) -> float:
    """Port of `Astar_Rec.m`."""
    bay, target_stacks = preprocess_bay(bay)
    n_containers = int(np.sum(bay != 0))
    if n_containers == bay.shape[1]:
        if best_incumbent > reloc + blocking_lower_bound(bay):
            best_incumbent = reloc + blocking_lower_bound(bay)
        return float(best_incumbent)

    t_size, s_size = bay.shape
    height = np.sum(bay != 0, axis=0).astype(int)
    ub_list = np.full(s_size, np.inf)
    target_stack = int(target_stacks[0])
    for s in range(s_size):
        if s != target_stack and int(height[s]) < t_size:
            new_bay = bay.copy()
            new_bay[t_size - int(height[s]) - 1, s] = bay[
                t_size - int(height[target_stack]), target_stack
            ]
            new_bay[t_size - int(height[target_stack]), target_stack] = 0
            ub_list[s] = reloc + 1 + run_heuristic_batch(
                new_bay, upper_bound_type, 1, rng=rng
            )
    sorted_stacks = np.argsort(ub_list)
    for s in sorted_stacks.tolist():
        if not np.isfinite(ub_list[s]):
            continue
        new_bay = bay.copy()
        new_bay[t_size - int(height[s]) - 1, s] = bay[
            t_size - int(height[target_stack]), target_stack
        ]
        new_bay[t_size - int(height[target_stack]), target_stack] = 0
        ub = float(ub_list[s])
        lb = (
            reloc
            + 1
            + blocking_lower_bound(new_bay)
            + rolling_lower_bound(new_bay, lower_bound_type)
        )
        if ub > lb and lb < best_incumbent:
            if best_incumbent > ub:
                best_incumbent = ub
            best_incumbent = _astar_rec(
                new_bay,
                lower_bound_type=lower_bound_type,
                upper_bound_type=upper_bound_type,
                best_incumbent=best_incumbent,
                reloc=reloc + 1,
                rng=rng,
            )
    return float(best_incumbent)
