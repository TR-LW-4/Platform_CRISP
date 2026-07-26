"""
Branching rules (Section 5 of the paper) used to prune symmetric or
dominated moves from the A*/IDA* search fringe. Applying any subset of
these rules preserves completeness (Propositions 1-4 of the paper): every
disallowed move is guaranteed to be reachable through an equivalent,
still-allowed ordering of moves, so no optimal solution is ever lost.

Rules implemented
------------------
0. Move reversal prevention (Section 5.1, always applied): never undo the
   immediately preceding move.
1/2. Unrelated move symmetry breaking (Section 5.2): two successive moves
   that touch entirely disjoint stacks can be reordered without changing
   the resulting layout; only one of the two orderings (chosen via a fixed
   ``<``/``>`` relation on the "from" stack) is explored.
     - "direct": only the immediately preceding move is examined (Rule 1).
     - "successive": the whole move history is examined, walking backwards
       until a move that touches either of the current move's stacks is
       found (Rule 2).
3/4. Transitive move avoidance (Section 5.3): moving a container from a to
   b and later from b to c is dominated by moving it directly from a to c;
   the second move of such a pair is disallowed.
     - "direct": only the immediately preceding move is examined (Rule 3).
     - "successive": the whole history is examined the same way as for the
       unrelated rule (Rule 4).
5. Empty stack symmetry breaking (Section 5.4): when >= 2 stacks are empty,
   only the lowest-indexed empty stack may receive a container.

This module is self-contained (only depends on ``core.py`` in this same
package) and does not import from any other algorithm.
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
