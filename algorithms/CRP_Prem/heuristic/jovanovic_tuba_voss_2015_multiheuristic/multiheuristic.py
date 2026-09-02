"""
48-combination enumeration for JovanovicTubaVoss2015MultiHeuristic.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from itertools import product
from typing import List, Optional, Sequence, Tuple

from .core import (
    Move,
    Stacks,
    best_destination,
    clone_stacks,
    not_well_located_refs,
    select_next_block,
)
from .correction import correct_moves
from .deadlock import resolve_deadlock
from .filling import FILLING_HEURISTICS, apply_filling
from .relocation import HEURISTICS as HW_SET, DeadlockSignal, relocate_and_place

HB_SET: Tuple[str, str] = ("descending", "lookahead")
HS_SET: Tuple[str, str] = ("w", "w_hat")
HF_SET: Tuple[str, ...] = FILLING_HEURISTICS

Combo = Tuple[str, str, str, str]


@dataclass
class RunResult:
    moves: List[Move] = field(default_factory=list)
    solved: bool = False
    remaining_not_well_located: int = 0
    combo: Optional[Combo] = None


def _well_locate_one(
    stacks: Stacks,
    ref: Tuple[int, int],
    p_expected: int,
    s_star: int,
    heuristic: str,
    max_tiers: int,
    moves: List[Move],
    rng: random.Random,
    max_retries: int = 50,
) -> bool:
    src, t = ref
    for _ in range(max_retries):
        if t >= len(stacks[src]) or stacks[src][t] != p_expected:
            return False  # defensive: state shifted unexpectedly, bail out safely
        try:
            relocate_and_place(stacks, ref, s_star, heuristic, max_tiers, moves)
            return True
        except DeadlockSignal as sig:
            if not resolve_deadlock(stacks, moves, sig.stuck_ref, sig.forbidden, max_tiers, rng):
                return False
    return False


def run_once(
    stacks_init: Stacks,
    max_tiers: int,
    hb: str,
    hs: str,
    hw: str,
    hf: str,
    rng_seed: int,
    safe_slack: int = 1,
) -> RunResult:
    stacks = clone_stacks(stacks_init)
    moves: List[Move] = []
    rng = random.Random(rng_seed)
    use_lookahead = hb == "lookahead"
    use_w_hat = hs == "w_hat"

    n_total = sum(len(v) for v in stacks_init.values())
    max_outer_iters = 4 * (n_total + 5)

    for _ in range(max_outer_iters):
        selection = select_next_block(stacks, use_lookahead=use_lookahead, max_tiers=max_tiers)
        if selection is None:
            break
        ref, p = selection
        s_star = best_destination(stacks, ref, use_improved=use_w_hat, max_tiers=max_tiers)
        if s_star is None:
            break
        ok = _well_locate_one(stacks, ref, p, s_star, hw, max_tiers, moves, rng)
        if not ok:
            break
        apply_filling(hf, stacks, touched=[s_star], max_tiers=max_tiers, moves=moves, safe_slack=safe_slack)

    moves = correct_moves(moves)
    remaining = len(not_well_located_refs(stacks))
    return RunResult(moves=moves, solved=(remaining == 0), remaining_not_well_located=remaining, combo=(hb, hs, hw, hf))


def all_combos() -> List[Combo]:
    return [(hb, hs, hw, hf) for hb, hs, hw, hf in product(HB_SET, HS_SET, HW_SET, HF_SET)]


def run_multiheuristic(
    stacks_init: Stacks,
    max_tiers: int,
    rng_seed: int = 0,
    safe_slack: int = 1,
    combos: Optional[Sequence[Combo]] = None,
) -> RunResult:
    """Algorithm of Sec. 4.3: run every combination, keep the best solution."""
    chosen = list(combos) if combos is not None else all_combos()
    best: Optional[Tuple[Tuple[int, int, int], RunResult]] = None
    for i, (hb, hs, hw, hf) in enumerate(chosen):
        res = run_once(stacks_init, max_tiers, hb, hs, hw, hf, rng_seed=rng_seed + i, safe_slack=safe_slack)
        key = (0 if res.solved else 1, res.remaining_not_well_located, len(res.moves))
        if best is None or key < best[0]:
            best = (key, res)
    assert best is not None  # all_combos() is never empty
    return best[1]
