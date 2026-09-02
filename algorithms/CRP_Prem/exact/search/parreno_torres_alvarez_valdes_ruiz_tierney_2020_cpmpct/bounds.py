"""
Lower bounds for ParrenoTorresAlvarezValdesRuizTierney2020CPMPCT.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List

from .core import Stacks, blocking_tiers
from .crane_time import CraneParams, min_move_time, move_time


def lower_bound_moves(stacks: Stacks) -> int:
    """
    Fast lower bound on number of relocation moves.
    """
    return len(blocking_tiers(stacks))


def lower_bound_time_ct0(stacks: Stacks, max_tiers: int, p: CraneParams) -> float:
    return float(lower_bound_moves(stacks)) * min_move_time(max_tiers, p)


def lower_bound_time_ct1(stacks: Stacks, max_tiers: int, p: CraneParams) -> float:
    """
    Paper-inspired LBct1:
    sum of minimum "touch-once" move times for initially blocking containers
    plus generic t_min for remaining needed moves.
    """
    lb = lower_bound_moves(stacks)
    tiers: List[int] = blocking_tiers(stacks)
    bcnt = len(tiers)
    acc = 0.0
    for h in tiers:
        # Best-case direct adjacent-stack relocation surrogate for one blocking box.
        acc += move_time(
            prev_dst_stack=0,   # surrogate; cancels in lower-bound spirit
            src_stack=0,
            src_level=h,
            dst_stack=1,
            dst_level=max_tiers,
            max_tiers=max_tiers,
            p=p,
        )
    return acc + max(0, lb - bcnt) * min_move_time(max_tiers, p)


def upper_bound_moves_from_time(best_time: float, max_tiers: int, p: CraneParams) -> int:
    if best_time <= 0:
        return 0
    tmin = min_move_time(max_tiers, p)
    return int(best_time / max(1e-9, tmin)) + 1
