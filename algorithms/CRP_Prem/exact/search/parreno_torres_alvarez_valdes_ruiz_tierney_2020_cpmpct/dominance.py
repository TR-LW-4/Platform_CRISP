from __future__ import annotations

from typing import Optional, Tuple

from .core import Move


def violates_direct_transitive(prev_move: Optional[Move], move: Move) -> bool:
    """
    Direct transitive rule (paper Proposition 8 flavor):
    avoid immediately moving from the just-used destination stack.
    """
    if prev_move is None:
        return False
    _, prev_dst = prev_move
    src, _ = move
    return src == prev_dst


def violates_same_group_rule(
    prev_move: Optional[Move],
    move: Move,
    prev_priority: Optional[int],
    moved_priority: int,
) -> bool:
    """
    Same-group symmetry adaptation used in CPMPCT:
    if previous move removed class p from stack s, do not move class p back to s
    immediately.
    """
    if prev_move is None or prev_priority is None:
        return False
    prev_src, _ = prev_move
    _, dst = move
    return (dst == prev_src) and (moved_priority == prev_priority)
