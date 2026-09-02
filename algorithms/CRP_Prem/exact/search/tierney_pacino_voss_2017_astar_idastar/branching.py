"""
Branching and symmetry rules for TierneyPacinoVoss2017AStarIDAStar.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Set, Tuple

from .core import Move, Stacks, legal_moves

Mode = str  # "off" | "direct" | "successive"
Direction = str  # "lt" | "gt"


def is_reversal(prev_move: Optional[Move], move: Move) -> bool:
    """Section 5.1: does ``move`` exactly undo ``prev_move``?"""
    if prev_move is None:
        return False
    f1, t1 = prev_move
    f2, t2 = move
    return t1 == f2 and t2 == f1


def _direction_ok(f1: int, f2: int, direction: Direction) -> bool:
    return f1 < f2 if direction == "lt" else f1 > f2


def violates_unrelated_rule(
    move: Move,
    history: Sequence[Move],
    direction: Direction,
    successive: bool,
) -> bool:
    """Rule 1 (``successive=False``) / Rule 2 (``successive=True``)."""
    if not history:
        return False
    f2, t2 = move
    lookback = history if successive else history[-1:]
    touched: Set[int] = set()
    for k in range(len(lookback) - 1, -1, -1):
        if touched & {f2, t2}:
            break
        fi, ti = lookback[k]
        unrelated = fi not in (f2, t2) and ti not in (f2, t2)
        if unrelated and touched.isdisjoint({fi, ti, f2, t2}):
            if not _direction_ok(fi, f2, direction):
                return True
        touched.add(fi)
        touched.add(ti)
    return False


def violates_transitive_rule(
    move: Move,
    history: Sequence[Move],
    successive: bool,
) -> bool:
    """Rule 3 (``successive=False``) / Rule 4 (``successive=True``)."""
    if not history:
        return False
    f2, t2 = move
    lookback = history if successive else history[-1:]
    touched: Set[int] = set()
    for k in range(len(lookback) - 1, -1, -1):
        if touched & {f2, t2}:
            break
        fi, ti = lookback[k]
        if ti == f2:
            return True
        touched.add(fi)
        touched.add(ti)
    return False


def violates_empty_stack_rule(stacks: Stacks, move: Move) -> bool:
    """Rule 5: only the lowest-indexed empty stack may receive a container."""
    empties = sorted(s for s, arr in stacks.items() if not arr)
    if len(empties) < 2:
        return False
    e_star = empties[0]
    _, dst = move
    return dst in empties and dst != e_star


def branches(
    stacks: Stacks,
    max_tiers: int,
    history: Sequence[Move],
    direction: Direction = "lt",
    unrelated_mode: Mode = "successive",
    transitive_mode: Mode = "successive",
    empty_stack_symmetry: bool = True,
) -> List[Move]:
    """
    ``branches(n)`` from Algorithms 1/2: every legal move out of ``stacks``
    that survives the enabled symmetry breaking / dominance rules.
    """
    prev_move = history[-1] if history else None
    out: List[Move] = []
    for mv in legal_moves(stacks, max_tiers):
        if is_reversal(prev_move, mv):
            continue
        if empty_stack_symmetry and violates_empty_stack_rule(stacks, mv):
            continue
        if transitive_mode != "off" and violates_transitive_rule(
            mv, history, successive=(transitive_mode == "successive")
        ):
            continue
        if unrelated_mode != "off" and violates_unrelated_rule(
            mv, history, direction, successive=(unrelated_mode == "successive")
        ):
            continue
        out.append(mv)
    return out
