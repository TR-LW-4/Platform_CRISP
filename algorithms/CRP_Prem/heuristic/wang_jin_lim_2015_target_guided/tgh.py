"""
Target-guided heuristic (TGH) main loop -- Algorithm 1 (Wang, Jin, Lim 2015).

::

    TGH(inilay)
    1  curlay = inilay
    2  g = G
    3  while g != 0
    4      mark clean containers with group label g as fixed
    5      conList = the set of candidate containers
    6      while conList != empty
    7          stkList = the set of candidate stacks
    8          determine the target container and stack from conList and stkList
    9          apply a giant move to fix the target container to the target stack
    10     g = g - 1

Termination (paper §4.5): only unfixed containers are ever chosen as targets
or relocation subjects; a fixed container is moved only transiently inside a
giant move's ``nslot == 0`` branch and is always restored before that giant
move returns. Hence each outer iteration strictly reduces the number of
unfixed containers, guaranteeing termination on any solvable instance.
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
