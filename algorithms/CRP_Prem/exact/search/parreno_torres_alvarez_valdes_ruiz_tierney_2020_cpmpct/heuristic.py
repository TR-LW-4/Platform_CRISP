"""
Upper-bound heuristic for ParrenoTorresAlvarezValdesRuizTierney2020CPMPCT.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .core import Move, Stacks, apply_move_inplace, clone_stacks, is_sorted, legal_moves, total_bad_overlaps
from .crane_time import CraneParams, move_time


def greedy_repair(
    stacks_init: Stacks,
    max_tiers: int,
    max_moves: int,
    p: CraneParams,
) -> Tuple[List[Move], float, bool]:
    """
    Lightweight BG/GG-style greedy repair for upper bound generation.
    """
    stacks = clone_stacks(stacks_init)
    moves: List[Move] = []
    t = 0.0
    prev_dst = -1

    for _ in range(max_moves):
        if is_sorted(stacks):
            return moves, t, True

        cands = legal_moves(stacks, max_tiers=max_tiers)
        if not cands:
            break

        best: Optional[Tuple[int, float, Move, int, int]] = None
        # tuple: (bad_after, dt, move, src_level, dst_level_after)
        for mv in cands:
            src, dst = mv
            if not stacks[src]:
                continue
            src_level = len(stacks[src])
            dst_level_after = len(stacks[dst]) + 1
            trial = clone_stacks(stacks)
            ok = apply_move_inplace(trial, src, dst, max_tiers=max_tiers)
            if not ok:
                continue
            bad_after = total_bad_overlaps(trial)
            dt = move_time(
                prev_dst_stack=prev_dst,
                src_stack=src,
                src_level=src_level,
                dst_stack=dst,
                dst_level=dst_level_after,
                max_tiers=max_tiers,
                p=p,
            )
            cand = (bad_after, dt, mv, src_level, dst_level_after)
            if best is None or cand[:2] < best[:2]:
                best = cand

        if best is None:
            break

        _, dt, (src, dst), _, _ = best
        ok = apply_move_inplace(stacks, src, dst, max_tiers=max_tiers)
        if not ok:
            break
        moves.append((src, dst))
        t += dt
        prev_dst = dst

    return moves, t, is_sorted(stacks)
