"""
A*/IDA* search loop for TierneyPacinoVoss2017AStarIDAStar.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from .bounds import lb_direct, lb_emo
from .branching import Direction, Mode, branches
from .core import (
    Move,
    Stacks,
    State,
    apply_move_inplace,
    clone_stacks,
    from_state,
    is_sorted,
    to_state,
)

_INF = 10 ** 9


@dataclass
class SolveResult:
    moves: List[Move]
    solved: bool
    proved_optimal: bool
    expanded_nodes: int
    lower_bound: int
    elapsed_s: float
    timed_out: bool = False


def _num_groups_of(stacks: Stacks) -> int:
    best = 0
    for arr in stacks.values():
        for p in arr:
            if p > best:
                best = p
    return best


def _heuristic(stacks: Stacks, max_tiers: int, use_emo: bool, num_groups: int) -> int:
    if use_emo:
        return lb_emo(stacks, max_tiers, num_groups)
    return lb_direct(stacks)


def _min_top_priority(stacks: Stacks) -> int:
    tops = [arr[-1] for arr in stacks.values() if arr]
    return min(tops) if tops else 0


# ==================================================================== #
#  A*  (Algorithm 1)                                                     #
# ==================================================================== #

def solve_astar(
    stacks_init: Stacks,
    max_tiers: int,
    time_limit_s: float = 60.0,
    use_emo: bool = True,
    unrelated_mode: Mode = "successive",
    transitive_mode: Mode = "successive",
    direction: Direction = "lt",
    empty_stack_symmetry: bool = True,
    use_memoization: bool = True,
    use_tie_breaking: bool = True,
    max_expansions: int = 500_000,
) -> SolveResult:
    start = time.perf_counter()
    num_groups = _num_groups_of(stacks_init)

    def h_of(stacks: Stacks) -> int:
        return _heuristic(stacks, max_tiers, use_emo, num_groups)

    init_state = to_state(stacks_init)
    if is_sorted(stacks_init):
        return SolveResult(
            moves=[], solved=True, proved_optimal=True,
            expanded_nodes=0, lower_bound=0,
            elapsed_s=time.perf_counter() - start,
        )

    counter = itertools.count()
    open_heap: List[Tuple] = []
    best_g: Dict[State, int] = {init_state: 0}
    history_of: Dict[State, Tuple[Move, ...]] = {init_state: tuple()}
    h_eff: Dict[State, int] = {init_state: h_of(stacks_init)}

    def push(state: State, g: int, parent_h: int) -> None:
        stacks = from_state(state)
        h_raw = h_of(stacks)
        h_use = max(h_raw, parent_h) if use_emo else h_raw
        h_eff[state] = h_use
        f = g + h_use
        tie1 = -g if use_tie_breaking else 0
        tie2 = _min_top_priority(stacks) if use_tie_breaking else 0
        heapq.heappush(open_heap, (f, tie1, tie2, next(counter), state))

    push(init_state, 0, h_eff[init_state])

    expanded = 0
    timed_out = False
    solved_state = None

    while open_heap:
        if (time.perf_counter() - start) >= time_limit_s:
            timed_out = True
            break
        if expanded >= max_expansions:
            timed_out = True
            break

        _f, _t1, _t2, _seq, state = heapq.heappop(open_heap)
        g = best_g.get(state)
        if g is None:
            continue

        stacks = from_state(state)
        if is_sorted(stacks):
            solved_state = state
            break

        expanded += 1
        history = history_of[state]
        parent_h = h_eff[state]

        for mv in branches(
            stacks, max_tiers, history,
            direction=direction,
            unrelated_mode=unrelated_mode,
            transitive_mode=transitive_mode,
            empty_stack_symmetry=empty_stack_symmetry,
        ):
            trial = clone_stacks(stacks)
            if not apply_move_inplace(trial, mv, max_tiers):
                continue
            new_state = to_state(trial)
            new_g = g + 1
            prev_g = best_g.get(new_state)
            if use_memoization and prev_g is not None and prev_g <= new_g:
                continue
            best_g[new_state] = new_g
            history_of[new_state] = history + (mv,)
            push(new_state, new_g, parent_h)

    elapsed = time.perf_counter() - start
    if solved_state is not None:
        moves = list(history_of[solved_state])
        return SolveResult(
            moves=moves, solved=True, proved_optimal=True,
            expanded_nodes=expanded, lower_bound=h_eff[init_state],
            elapsed_s=elapsed, timed_out=False,
        )
    return SolveResult(
        moves=[], solved=False, proved_optimal=False,
        expanded_nodes=expanded, lower_bound=h_eff[init_state],
        elapsed_s=elapsed, timed_out=timed_out,
    )


# ==================================================================== #
#  IDA*  (Algorithm 2)                                                   #
# ==================================================================== #

def solve_idastar(
    stacks_init: Stacks,
    max_tiers: int,
    time_limit_s: float = 60.0,
    use_emo: bool = True,
    unrelated_mode: Mode = "successive",
    transitive_mode: Mode = "successive",
    direction: Direction = "lt",
    empty_stack_symmetry: bool = True,
    depth_padding: int = 200,
) -> SolveResult:
    start = time.perf_counter()
    num_groups = _num_groups_of(stacks_init)

    def h_of(stacks: Stacks) -> int:
        return _heuristic(stacks, max_tiers, use_emo, num_groups)

    if is_sorted(stacks_init):
        return SolveResult(
            moves=[], solved=True, proved_optimal=True,
            expanded_nodes=0, lower_bound=0,
            elapsed_s=time.perf_counter() - start,
        )

    def timed_out_now() -> bool:
        return (time.perf_counter() - start) >= time_limit_s

    init_h = h_of(stacks_init)
    bound = init_h
    max_bound = init_h + max(0, depth_padding)

    path: List[Move] = []
    expanded = 0
    hit_time_limit = False

    def dfs(stacks: Stacks, g: int, bound_now: int, parent_h: int) -> Tuple[bool, int]:
        nonlocal expanded, hit_time_limit
        if timed_out_now():
            hit_time_limit = True
            return False, _INF
        if is_sorted(stacks):
            return True, g

        h_raw = h_of(stacks)
        h_use = max(h_raw, parent_h) if use_emo else h_raw
        f = g + h_use
        if f > bound_now:
            return False, f

        expanded += 1
        next_threshold = _INF
        history = tuple(path)

        for mv in branches(
            stacks, max_tiers, history,
            direction=direction,
            unrelated_mode=unrelated_mode,
            transitive_mode=transitive_mode,
            empty_stack_symmetry=empty_stack_symmetry,
        ):
            trial = clone_stacks(stacks)
            if not apply_move_inplace(trial, mv, max_tiers):
                continue
            path.append(mv)
            found, res = dfs(trial, g + 1, bound_now, h_use)
            if found:
                return True, res
            path.pop()
            if res < next_threshold:
                next_threshold = res
            if hit_time_limit:
                return False, _INF
        return False, next_threshold

    solved = False
    while bound <= max_bound and not timed_out_now():
        found, nxt = dfs(stacks_init, 0, bound, init_h)
        if found:
            solved = True
            break
        if hit_time_limit or nxt >= _INF:
            break
        bound = max(bound + 1, nxt)

    elapsed = time.perf_counter() - start
    return SolveResult(
        moves=list(path) if solved else [],
        solved=solved,
        proved_optimal=solved,
        expanded_nodes=expanded,
        lower_bound=init_h,
        elapsed_s=elapsed,
        timed_out=(not solved) and hit_time_limit,
    )
