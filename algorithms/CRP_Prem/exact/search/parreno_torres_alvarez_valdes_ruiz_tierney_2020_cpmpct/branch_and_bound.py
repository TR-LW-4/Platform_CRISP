"""
Branch-and-bound driver for ParrenoTorresAlvarezValdesRuizTierney2020CPMPCT.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .bounds import lower_bound_moves, lower_bound_time_ct0, lower_bound_time_ct1
from .core import Move, Stacks, apply_move_inplace, clone_stacks, is_sorted, legal_moves, total_bad_overlaps
from .crane_time import CraneParams, move_time
from .dominance import violates_direct_transitive, violates_same_group_rule
from .heuristic import greedy_repair


@dataclass
class BBSolveResult:
    moves: List[Move]
    crane_time: float
    solved: bool
    expanded_nodes: int
    elapsed_s: float


def solve_cpmpct_branch_and_bound(
    stacks_init: Stacks,
    max_tiers: int,
    p: CraneParams,
    time_limit_s: float,
    max_extra_depth: int = 100,
) -> BBSolveResult:
    t0 = time.perf_counter()
    expanded = 0
    best_moves: List[Move] = []
    best_time = float("inf")
    solved = False

    # Heuristic upper bound.
    ub_seed, ub_time, ub_ok = greedy_repair(
        stacks_init=stacks_init,
        max_tiers=max_tiers,
        max_moves=max(10, len([x for arr in stacks_init.values() for x in arr]) * 4),
        p=p,
    )
    if ub_ok:
        best_moves = ub_seed
        best_time = ub_time
        solved = True

    lb_depth = lower_bound_moves(stacks_init)
    max_depth = lb_depth + max(0, int(max_extra_depth))

    def timed_out() -> bool:
        return (time.perf_counter() - t0) >= time_limit_s

    def dfs(
        stacks: Stacks,
        depth: int,
        limit: int,
        acc_time: float,
        prev_move: Optional[Move],
        prev_priority: Optional[int],
        prev_dst_stack: int,
        path: List[Move],
    ) -> None:
        nonlocal expanded, best_moves, best_time, solved
        if timed_out():
            return
        expanded += 1

        if is_sorted(stacks):
            solved = True
            if acc_time < best_time:
                best_time = acc_time
                best_moves = list(path)
            return

        if depth >= limit:
            return

        # Time lower bounds.
        lb_t = max(
            lower_bound_time_ct0(stacks, max_tiers=max_tiers, p=p),
            lower_bound_time_ct1(stacks, max_tiers=max_tiers, p=p),
        )
        if acc_time + lb_t >= best_time:
            return

        cands = legal_moves(stacks, max_tiers=max_tiers)
        scored: List[Tuple[int, float, Move, int, int, int]] = []
        for mv in cands:
            src, dst = mv
            if not stacks[src]:
                continue
            moved_prio = int(stacks[src][-1])
            if violates_direct_transitive(prev_move, mv):
                continue
            if violates_same_group_rule(prev_move, mv, prev_priority, moved_prio):
                continue

            src_level = len(stacks[src])
            dst_level_after = len(stacks[dst]) + 1
            dt = move_time(
                prev_dst_stack=prev_dst_stack,
                src_stack=src,
                src_level=src_level,
                dst_stack=dst,
                dst_level=dst_level_after,
                max_tiers=max_tiers,
                p=p,
            )
            trial = clone_stacks(stacks)
            ok = apply_move_inplace(trial, src, dst, max_tiers=max_tiers)
            if not ok:
                continue
            bad = total_bad_overlaps(trial)
            scored.append((bad, int(dt * 1000.0), mv, moved_prio, src_level, dst_level_after))

        scored.sort(key=lambda x: (x[0], x[1]))
        for _, _, mv, moved_prio, _, _ in scored:
            src, dst = mv
            src_level = len(stacks[src])
            dst_level_after = len(stacks[dst]) + 1
            dt = move_time(
                prev_dst_stack=prev_dst_stack,
                src_stack=src,
                src_level=src_level,
                dst_stack=dst,
                dst_level=dst_level_after,
                max_tiers=max_tiers,
                p=p,
            )
            if acc_time + dt >= best_time:
                continue
            ok = apply_move_inplace(stacks, src, dst, max_tiers=max_tiers)
            if not ok:
                continue
            path.append((src, dst))
            dfs(
                stacks=stacks,
                depth=depth + 1,
                limit=limit,
                acc_time=acc_time + dt,
                prev_move=mv,
                prev_priority=moved_prio,
                prev_dst_stack=dst,
                path=path,
            )
            # undo move
            ok2 = apply_move_inplace(stacks, dst, src, max_tiers=max_tiers)
            path.pop()
            if not ok2:
                return
            if timed_out():
                return

    root = clone_stacks(stacks_init)
    for limit in range(lb_depth, max_depth + 1):
        if timed_out():
            break
        dfs(
            stacks=root,
            depth=0,
            limit=limit,
            acc_time=0.0,
            prev_move=None,
            prev_priority=None,
            prev_dst_stack=-1,
            path=[],
        )
        if solved and best_moves and len(best_moves) <= limit:
            # Found best-known solution at current depth; continue for possible
            # lower-time same-or-higher-depth improvements until timeout.
            pass

    return BBSolveResult(
        moves=best_moves,
        crane_time=best_time if best_time < float("inf") else 0.0,
        solved=solved,
        expanded_nodes=expanded,
        elapsed_s=time.perf_counter() - t0,
    )
