"""
Type-ordering helpers for LeeHsu2007NetworkFlowIP.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .core import Move, Stacks, clone_stacks


class OrderingFailure(Exception):
    pass


def order_segment_moves(stacks: Stacks, moves: List[Move], max_tiers: int) -> List[Move]:
    """Mutates ``stacks`` in place to reflect the state after the segment.

    A movement arc cr[t,s,z,c] always refers to the container on top of
    stack s *as of time point t* (the segment's starting state), not to
    whatever might already have landed on s from another move within the
    same segment. So when a stack already has a container on top *and*
    some pending move wants to take it away, any *other* pending move that
    would land a new container on that same stack must wait -- otherwise
    the later-arriving container would wrongly bury the one that was
    actually meant to leave. A stack that starts this segment empty has no
    such "original top" to protect, so moves landing on it (e.g. the relay
    moves inserted below to break a cycle) are never deferred on that
    account.
    """
    remaining = list(dict.fromkeys(moves))  # de-duplicate, preserve order
    ordered: List[Move] = []
    guard = 0
    max_guard = 20 * (len(moves) + 5)

    while remaining:
        guard += 1
        if guard > max_guard:
            raise OrderingFailure("Exceeded iteration guard while ordering segment moves.")

        pending_sources = {s for s, _ in remaining}
        progressed = False
        for i, (src, dst) in enumerate(remaining):
            if src not in stacks or dst not in stacks:
                remaining.pop(i)
                progressed = True
                break
            if stacks[dst] and dst in pending_sources:
                continue  # dst's original top must be sent away first
            if stacks[src] and len(stacks[dst]) < max_tiers:
                stacks[dst].append(stacks[src].pop())
                ordered.append((src, dst))
                remaining.pop(i)
                progressed = True
                break

        if progressed:
            continue

        # No move is immediately executable -> a cycle blocks all of them.
        # Break it by rerouting the first blocked move through a spare stack.
        src, dst = remaining[0]
        involved = {s for mv in remaining for s in mv}
        spare = next(
            (s for s in stacks if s not in involved and len(stacks[s]) < max_tiers),
            None,
        )
        if spare is None:
            spare = next((s for s in stacks if s != src and len(stacks[s]) < max_tiers), None)
        if spare is None:
            raise OrderingFailure("No spare stack with room available to break a movement cycle.")

        remaining[0] = (src, spare)
        remaining.append((spare, dst))

    return ordered


def reconstruct_move_sequence(
    stacks_init: Stacks,
    segment_moves: Dict[int, List[Move]],
    max_tiers: int,
) -> "tuple[List[Move], Stacks]":
    """
    Applies each segment's moves (in segment-index order) via
    ``order_segment_moves`` and returns the concatenated, fully ordered
    move list along with the resulting stack layout.
    """
    st = clone_stacks(stacks_init)
    all_moves: List[Move] = []
    for t in sorted(segment_moves.keys()):
        moves = segment_moves[t]
        if not moves:
            continue
        ordered = order_segment_moves(st, moves, max_tiers)
        all_moves.extend(ordered)
    return all_moves, st
