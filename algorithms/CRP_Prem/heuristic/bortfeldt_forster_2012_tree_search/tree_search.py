"""
Tree-search driver for BortfeldtForster2012TreeSearch.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .compound_moves import (
    CompoundMove,
    LowerBoundFn,
    generate_extra_compound_moves,
    generate_normal_compound_moves,
)
from .core import (
    Move,
    Stacks,
    is_final_layout,
)


@dataclass
class SearchResult:
    moves: List[Move]
    solved: bool
    n_nodes_expanded: int
    n_compound_moves_evaluated: int
    n_normal_compound_moves: int
    n_extra_compound_moves: int
    time_seconds: float
    lower_bound_init: int
    aborted_by_time: bool


@dataclass
class _SearchState:
    """Mutable state shared across recursive calls."""
    best_moves: List[Move] = field(default_factory=list)
    best_count: int = 10**9
    solved: bool = False
    n_nodes: int = 0
    n_cm: int = 0
    n_normal: int = 0
    n_extra: int = 0
    deadline: float = 0.0
    optimal_target: int = 0  # once best_count == this, we can early-return


def perform_tree_search(
    initial_stacks: Stacks,
    *,
    max_tiers: int,
    num_groups: int,
    lb_fn: LowerBoundFn,
    n_succ: int = 5,
    pub: float = 1.75,
    time_limit_s: float = 20.0,
    use_filtering: bool = True,
    truncate_to_single_move: bool = False,
    sort_output: bool = True,
) -> SearchResult:
    """
    Run the BF-2012 heuristic tree search on ``initial_stacks``.
    """
    t0 = time.perf_counter()
    lb_init = lb_fn(initial_stacks, num_groups, max_tiers)
    state = _SearchState(
        best_moves=[],
        best_count=10**9,
        solved=False,
        n_nodes=0,
        n_cm=0,
        n_normal=0,
        n_extra=0,
        deadline=t0 + max(0.0, float(time_limit_s)),
        optimal_target=lb_init,
    )

    # Fast exit: initial layout already final.
    if is_final_layout(initial_stacks):
        return SearchResult(
            moves=[],
            solved=True,
            n_nodes_expanded=0,
            n_compound_moves_evaluated=0,
            n_normal_compound_moves=0,
            n_extra_compound_moves=0,
            time_seconds=time.perf_counter() - t0,
            lower_bound_init=lb_init,
            aborted_by_time=False,
        )

    _recurse(
        stacks=initial_stacks,
        imported_moves=[],
        state=state,
        max_tiers=max_tiers,
        num_groups=num_groups,
        lb_fn=lb_fn,
        n_succ=n_succ,
        pub=pub,
        lb_init=lb_init,
        use_filtering=use_filtering,
        truncate_to_single_move=truncate_to_single_move,
        sort_output=sort_output,
    )

    elapsed = time.perf_counter() - t0
    return SearchResult(
        moves=list(state.best_moves),
        solved=state.solved,
        n_nodes_expanded=state.n_nodes,
        n_compound_moves_evaluated=state.n_cm,
        n_normal_compound_moves=state.n_normal,
        n_extra_compound_moves=state.n_extra,
        time_seconds=elapsed,
        lower_bound_init=lb_init,
        aborted_by_time=(time.perf_counter() >= state.deadline and not state.solved),
    )


def _recurse(
    *,
    stacks: Stacks,
    imported_moves: List[Move],
    state: _SearchState,
    max_tiers: int,
    num_groups: int,
    lb_fn: LowerBoundFn,
    n_succ: int,
    pub: float,
    lb_init: int,
    use_filtering: bool,
    truncate_to_single_move: bool,
    sort_output: bool,
) -> None:
    # Time and optimality abort.
    if time.perf_counter() >= state.deadline:
        return
    if state.solved and state.best_count <= state.optimal_target:
        return

    state.n_nodes += 1

    if is_final_layout(stacks):
        if len(imported_moves) < state.best_count:
            state.best_count = len(imported_moves)
            state.best_moves = list(imported_moves)
            state.solved = True
        return

    normal_cms: List[CompoundMove] = generate_normal_compound_moves(
        stacks=stacks,
        imported_moves=imported_moves,
        max_tiers=max_tiers,
        num_groups=num_groups,
        n_succ=n_succ,
        pub=pub,
        best_move_count=state.best_count,
        lb_init=lb_init,
        lb_fn=lb_fn,
        use_filtering=use_filtering,
        truncate_to_single_move=truncate_to_single_move,
        sort_output=sort_output,
    )
    if normal_cms:
        state.n_normal += len(normal_cms)
        state.n_cm += len(normal_cms)
        cms = normal_cms
    else:
        extra_cms = generate_extra_compound_moves(
            stacks=stacks,
            imported_moves=imported_moves,
            max_tiers=max_tiers,
            num_groups=num_groups,
            n_succ=n_succ,
            pub=pub,
            best_move_count=state.best_count,
            lb_init=lb_init,
            lb_fn=lb_fn,
            use_filtering=use_filtering,
            truncate_to_single_move=truncate_to_single_move,
            sort_output=sort_output,
        )
        state.n_extra += len(extra_cms)
        state.n_cm += len(extra_cms)
        cms = extra_cms

    if not cms:
        return

    for cm in cms:
        if time.perf_counter() >= state.deadline:
            return
        if state.solved and state.best_count <= state.optimal_target:
            return
        new_imported = imported_moves + cm.moves
        _recurse(
            stacks=cm.result,
            imported_moves=new_imported,
            state=state,
            max_tiers=max_tiers,
            num_groups=num_groups,
            lb_fn=lb_fn,
            n_succ=n_succ,
            pub=pub,
            lb_init=lb_init,
            use_filtering=use_filtering,
            truncate_to_single_move=truncate_to_single_move,
            sort_output=sort_output,
        )
