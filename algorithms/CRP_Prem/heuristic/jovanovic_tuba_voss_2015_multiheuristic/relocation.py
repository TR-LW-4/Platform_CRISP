"""
Stage 3 (Sec. 3.3): relocating the blocking blocks necessary to well-locate
a container, and the three competing "which stack to relocate to" heuristics
(the Hw set): TLP, LPI, MinMax.

* TLP (Zhang 2000): relocate to the stack with the fewest used tiers
  (balance the bay).
* LPI (Exposito-Izquierdo et al. 2012): relocate to the stack whose current
  top is *not well located* and has the highest due-date value (i.e. the
  least urgent "damage"); a stack whose top is well located (or is empty)
  is always preferred over any not-well-located option.
* MinMax (Caserta, Schwarze, Voss 2011b / Unluyurt & Aydin 2012): if the
  relocated block would become well located on some candidate stack, prefer
  the tightest fit (smallest top value still >= the block's own value);
  otherwise prefer the stack with the smallest maximal due-date value
  currently present (avoid spreading badly-mixed groups around). The paper
  cites this heuristic rather than fully specifying it; this is the
  standard formulation used across the BRP/PMP literature it references.

Per Sec. 3.3, when the source and destination stack coincide (the block c
being well-located must be temporarily parked out of its own stack), "the
heuristic function should be used with inverse values": we implement this
by negating the ranking key, and also avoid ever leaving a stack completely
full when a heuristic other than TLP is used ("reaching the top tier of a
stack should also be avoided").
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .core import Stacks, f_count, is_well_located_stack

Move = Tuple[int, int]

HEURISTICS = ("TLP", "LPI", "MinMax")


def _tlp_key(stacks: Stacks, s: int, p: int) -> Tuple:
    return (len(stacks[s]),)


def _lpi_key(stacks: Stacks, s: int, p: int) -> Tuple:
    arr = stacks[s]
    if not arr or is_well_located_stack(arr):
        return (0, 0)
    return (1, -arr[-1])


def _minmax_key(stacks: Stacks, s: int, p: int) -> Tuple:
    arr = stacks[s]
    if not arr:
        return (0, 0)
    top = arr[-1]
    if top >= p:
        return (0, top)
    return (1, max(arr))


_KEY_FUNCS = {"TLP": _tlp_key, "LPI": _lpi_key, "MinMax": _minmax_key}


def hw_key(heuristic: str, stacks: Stacks, s: int, p: int, inverse: bool) -> Tuple:
    key = _KEY_FUNCS[heuristic](stacks, s, p)
    if inverse:
        key = tuple(-x for x in key)
    return key


def choose_temp_stack(
    stacks: Stacks,
    forbidden: Tuple[int, ...],
    p: int,
    heuristic: str,
    max_tiers: int,
    inverse: bool = False,
) -> Optional[int]:
    candidates = [s for s in stacks if s not in forbidden and len(stacks[s]) < max_tiers]
    if not candidates:
        return None
    if heuristic == "TLP":
        candidates.sort(key=lambda s: (hw_key(heuristic, stacks, s, p, inverse), s))
        return candidates[0]
    # LPI / MinMax: soft-avoid leaving the stack completely full.
    candidates.sort(
        key=lambda s: (
            1 if len(stacks[s]) + 1 >= max_tiers else 0,
            hw_key(heuristic, stacks, s, p, inverse),
            s,
        )
    )
    return candidates[0]


def _move_top(stacks: Stacks, src: int, dst: int, max_tiers: int) -> bool:
    if src == dst or not stacks[src] or len(stacks[dst]) >= max_tiers:
        return False
    stacks[dst].append(stacks[src].pop())
    return True


class DeadlockSignal(Exception):
    """Raised when stage 3 cannot find any valid destination for a blocker."""

    def __init__(self, stuck_ref, forbidden):
        super().__init__("no valid relocation destination")
        self.stuck_ref = stuck_ref
        self.forbidden = forbidden


def relocate_and_place(
    stacks: Stacks,
    ref: Tuple[int, int],
    s_star: int,
    heuristic: str,
    max_tiers: int,
    moves: List[Move],
) -> None:
    """
    Physically well-locate the container at ``ref`` onto stack ``s_star``,
    per Sec. 3.3's ordering rule (at each step relocate whichever of the two
    exposed blockers -- above c in its own stack, or blocking the landing
    slot in s* -- needs to move; identical ordering rule to the one already
    embedded for Exposito-Izquierdo (2012), which this paper does not
    modify), using ``heuristic`` (one of :data:`HEURISTICS`) to choose each
    blocker's temporary stack.

    Raises :class:`DeadlockSignal` if some blocker has nowhere to go (the
    caller is expected to invoke the deadlock-avoidance mechanism and
    retry).
    """
    src, t = ref
    p = stacks[src][t]

    # NOTE: priority values need not be unique (the paper explicitly allows
    # ties), so the target container must be tracked by *position*, not by
    # comparing values. Since we only ever pop from the top of a stack, the
    # index ``t`` of the target within ``stacks[src]`` never shifts while it
    # is still present; "exposed at the top" means ``len(stacks[src])-1==t``.

    if src == s_star:
        # Phase 1: clear everything currently sitting above the target.
        while len(stacks[src]) - 1 > t:
            top_val = stacks[src][-1]
            tm = choose_temp_stack(stacks, forbidden=(src,), p=top_val, heuristic=heuristic,
                                    max_tiers=max_tiers, inverse=False)
            if tm is None or not _move_top(stacks, src, tm, max_tiers):
                raise DeadlockSignal((src, len(stacks[src]) - 1), (src,))
            moves.append((src, tm))

        # Target is now exposed at the top; park it (inverse heuristic per
        # Sec. 3.3's "special care" rule for the s = s* case).
        tmp = choose_temp_stack(stacks, forbidden=(src,), p=p, heuristic=heuristic,
                                 max_tiers=max_tiers, inverse=True)
        if tmp is None or not _move_top(stacks, src, tmp, max_tiers):
            raise DeadlockSignal(ref, (src,))
        moves.append((src, tmp))

        # Phase 2: clear whatever now blocks the target's landing spot.
        while f_count(stacks, src, p) > 0:
            top_val = stacks[src][-1]
            tm = choose_temp_stack(stacks, forbidden=(src, tmp), p=top_val,
                                    heuristic=heuristic, max_tiers=max_tiers, inverse=False)
            if tm is None or not _move_top(stacks, src, tm, max_tiers):
                raise DeadlockSignal((src, len(stacks[src]) - 1), (src, tmp))
            moves.append((src, tm))

        # Bring the target back to its now-well-located landing spot.
        if not _move_top(stacks, tmp, src, max_tiers):
            raise DeadlockSignal((tmp, len(stacks[tmp]) - 1), (tmp, src))
        moves.append((tmp, src))
        return

    while True:
        still_covered = len(stacks[src]) - 1 > t
        need_remove = f_count(stacks, s_star, p)
        if not still_covered and need_remove <= 0:
            break

        o_val = stacks[src][-1] if still_covered else None
        m_val = stacks[s_star][-1] if need_remove > 0 else None
        if o_val is None and m_val is None:
            raise DeadlockSignal(ref, (src, s_star))

        move_from_src = o_val is not None and (m_val is None or o_val < m_val)
        if move_from_src:
            tm = choose_temp_stack(stacks, forbidden=(src, s_star), p=o_val,
                                    heuristic=heuristic, max_tiers=max_tiers, inverse=False)
            if tm is None or not _move_top(stacks, src, tm, max_tiers):
                raise DeadlockSignal((src, len(stacks[src]) - 1), (src, s_star))
            moves.append((src, tm))
        else:
            tm = choose_temp_stack(stacks, forbidden=(src, s_star), p=m_val,
                                    heuristic=heuristic, max_tiers=max_tiers, inverse=False)
            if tm is None or not _move_top(stacks, s_star, tm, max_tiers):
                raise DeadlockSignal((s_star, len(stacks[s_star]) - 1), (src, s_star))
            moves.append((s_star, tm))

    if not _move_top(stacks, src, s_star, max_tiers):
        raise DeadlockSignal(ref, (src, s_star))
    moves.append((src, s_star))
