"""
Target-guided construction loop for WangJinLim2015TGH.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .core import (
    InfeasibleInstance,
    auto_fix_clean_prefix,
    candidate_containers,
    candidate_stacks,
    giant_move,
    select_target_pair,
)
from .state import Move, State, is_solved


@dataclass
class TGHResult:
    moves: List[Move]
    solved: bool
    infeasible_reason: str = ""


def tgh_solve(state: State, num_groups: int, max_moves: int = 10**6) -> TGHResult:
    moves: List[Move] = []
    try:
        g = num_groups
        while g != 0:
            auto_fix_clean_prefix(state, g)
            con_list = candidate_containers(state, g)
            while con_list:
                if len(moves) > max_moves:
                    return TGHResult(moves=moves, solved=False, infeasible_reason="move budget exceeded")
                stk_list = candidate_stacks(state, g)
                if not stk_list:
                    return TGHResult(moves=moves, solved=False, infeasible_reason=f"no candidate stack for group {g}")
                target, target_stack = select_target_pair(state, con_list, stk_list)
                moves.extend(giant_move(state, target, target_stack))
                con_list = candidate_containers(state, g)
            g -= 1
    except InfeasibleInstance as exc:
        return TGHResult(moves=moves, solved=False, infeasible_reason=str(exc))

    return TGHResult(moves=moves, solved=is_solved(state), infeasible_reason="")
