"""
Solve driver for TanakaTierney2018IDBB.

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
from typing import List, Optional

from .bridge import run_idbb
from .core import Move, Stacks, apply_moves, count_bad_overlaps, is_fully_well_located


@dataclass
class SolveResult:
    moves: List[Move] = field(default_factory=list)
    solved: bool = False
    proved_optimal: bool = False
    n_relocation: Optional[int] = None
    timed_out: bool = False
    remaining_bad_overlaps: int = 0
    elapsed_s: float = 0.0
    error: Optional[str] = None


def solve_idbb(
    stacks_init: Stacks,
    max_tiers: int,
    time_limit_s: float = 30.0,
) -> SolveResult:
    t0 = time.perf_counter()

    if is_fully_well_located(stacks_init):
        return SolveResult(
            moves=[], solved=True, proved_optimal=True, n_relocation=0,
            elapsed_s=time.perf_counter() - t0,
        )

    vendor_result = run_idbb(stacks_init, max_tiers=max_tiers, time_limit_s=time_limit_s)

    if vendor_result.error is not None:
        return SolveResult(error=vendor_result.error, elapsed_s=time.perf_counter() - t0)

    final_stacks = apply_moves(stacks_init, vendor_result.moves, max_tiers)
    bad = count_bad_overlaps(final_stacks)
    total_before = sum(len(v) for v in stacks_init.values())
    total_after = sum(len(v) for v in final_stacks.values())
    solved = vendor_result.solved and bad == 0 and total_before == total_after

    return SolveResult(
        moves=vendor_result.moves,
        solved=solved,
        proved_optimal=vendor_result.proved_optimal and solved,
        n_relocation=vendor_result.n_relocation,
        timed_out=vendor_result.timed_out,
        remaining_bad_overlaps=bad,
        elapsed_s=time.perf_counter() - t0,
    )
