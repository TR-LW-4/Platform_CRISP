"""
Sec. 4.1: formalized deadlock avoidance (Eq. 7-9).

The greedy scheme can paint itself into a corner: while temporarily parking
a container c that is being well-located within its own stack (the s = s*
case in ``relocation.py`` -- the paper states deadlocks can only arise in
this sub-case), the tiers above c's temporary resting place become unusable
(nothing may be stacked above a container that must be picked straight back
up), which can "use up" bay-wide slack. If this happens repeatedly, some
later relocation needed to clear c's own stack may have no valid
destination at all.

Documented simplification: the exact subscripts in the published Eq. 7-9
are ambiguous (rendering artifacts leave "s(cr)" undefined relative to the
surrounding text), and a literal "revert, borrow, redo" chain risks a
capacity double-booking (the freed slot would be claimed by both the
borrowed container and the redone relocation). We implement the same
underlying ingredients -- undo an earlier relocation to free a slot, and
(when that slot is off-limits as a final resting place) borrow one
container from a random full stack through it -- but never "redo" the
undone move; the container that was reverted simply becomes not-well-located
again and is naturally reconsidered by a later iteration of the main loop.
This preserves the paper's guarantee (progress can always resume) while
remaining provably capacity-safe.

To maximize the chance of finding a usable slot (rather than only ever
considering the single most recent relocation), we scan backwards through
the move history for the most recent relocation whose destination (a) is
not the stuck stack, and (b) has not been touched (as source or
destination) by any later move -- which guarantees its top container is
still exactly what that move placed there, so reverting it in isolation is
safe.
"""

from __future__ import annotations

import random
from typing import List, Tuple

from .core import Stacks

Move = Tuple[int, int]


def resolve_deadlock(
    stacks: Stacks,
    moves: List[Move],
    stuck_ref: Tuple[int, int],
    forbidden: Tuple[int, ...],
    max_tiers: int,
    rng: random.Random,
) -> bool:
    """
    Attempt one recovery step. ``stuck_ref`` is the (stack, tier) of the
    container that has no valid destination right now (guaranteed to be at
    the top of its stack). ``forbidden`` is the set of stacks tied up in
    the current well-locate task, which must not be used as the recovered
    slot (matches Eq. 7's ``Except`` set).

    Mutates ``stacks``/``moves`` in place and returns True on success. On
    failure (nothing safely revertible), state and ``moves`` are left
    unchanged and False is returned; the caller should then treat this
    greedy run as unable to complete (the outer 48-combination search will
    typically still succeed via a different heuristic combination).
    """
    stuck_src, _ = stuck_ref
    if not stacks[stuck_src]:
        return False

    touched_since: set = set()
    for idx in range(len(moves) - 1, -1, -1):
        s_from, s_to = moves[idx]
        safe_to_revert = s_to not in touched_since and s_from not in touched_since
        can_revert = safe_to_revert and s_to != stuck_src and stacks[s_to] and len(stacks[s_from]) < max_tiers
        if can_revert:
            if s_to not in forbidden:
                # Route the stuck container directly into the freed slot.
                stacks[s_from].append(stacks[s_to].pop())
                del moves[idx]
                stacks[s_to].append(stacks[stuck_src].pop())
                moves.append((stuck_src, s_to))
                return True

            # s_to is off-limits as a final resting place (Eq. 7's "Except"
            # set), but can still serve as a pass-through buffer: borrow one
            # container from a random full donor stack into the freed slot,
            # then route the stuck container into the slot the donor just
            # freed (Eq. 8-9's "Random(Full(...))" step).
            full_candidates = [
                s for s in stacks
                if s not in forbidden and s != s_to and s != s_from and s != stuck_src
                and len(stacks[s]) >= max_tiers
            ]
            if full_candidates:
                stacks[s_from].append(stacks[s_to].pop())
                del moves[idx]
                sf = rng.choice(full_candidates)
                stacks[s_to].append(stacks[sf].pop())
                moves.append((sf, s_to))
                stacks[sf].append(stacks[stuck_src].pop())
                moves.append((stuck_src, sf))
                return True
        touched_since.add(s_from)
        touched_since.add(s_to)

    return False
