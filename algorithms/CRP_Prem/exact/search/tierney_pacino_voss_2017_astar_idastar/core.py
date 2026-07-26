"""
Shared bay/move representation for the Tierney, Pacino & Voß (2017) A*/IDA*
pre-marshalling embedding.

Paper:
K. Tierney, D. Pacino, S. Voß, "Solving the Pre-Marshalling Problem to
Optimality with A* and IDA*", Flexible Services and Manufacturing Journal
29(2), 223-259, 2017.

Stack/priority convention (Section 2): ``pst`` is the priority of the
container in stack ``s`` at tier ``t``, smaller values leave the bay earlier
and therefore belong nearer the *top* of the stack in a mis-overlay-free
layout. A bay has no mis-overlays iff, for every stack, priorities are
non-increasing from bottom to top. This matches the rest of the platform's
``core.yard.Stack.is_sorted_by_priority`` convention.

This module is intentionally self-contained: it does not import from any
other algorithm package.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

Stacks = Dict[int, List[int]]     # stack_idx -> priorities, bottom .. top
Move = Tuple[int, int]            # (from_stack, to_stack)
State = Tuple[Tuple[int, ...], ...]


def clone_stacks(stacks: Stacks) -> Stacks:
    return {k: list(v) for k, v in stacks.items()}


def to_state(stacks: Stacks) -> State:
    """Canonical, hashable representation keyed by sorted stack index."""
    return tuple(tuple(stacks[s]) for s in sorted(stacks))


def from_state(state: State) -> Stacks:
    return {i: list(arr) for i, arr in enumerate(state)}


def num_stacks(stacks: Stacks) -> int:
    return len(stacks)


def apply_move_inplace(stacks: Stacks, move: Move, max_tiers: int) -> bool:
    """Apply ``move = (src, dst)`` in place. Returns True iff it was legal."""
    src, dst = move
    if src == dst:
        return False
    if src not in stacks or dst not in stacks:
        return False
    if not stacks[src] or len(stacks[dst]) >= max_tiers:
        return False
    stacks[dst].append(stacks[src].pop())
    return True


def undo_move_inplace(stacks: Stacks, move: Move) -> None:
    src, dst = move
    stacks[src].append(stacks[dst].pop())


def first_bad_tier(arr: List[int]) -> int:
    """
    0-based index (bottom = 0) of the lowest mis-overlaid container in
    ``arr``. Uses the paper's transitive notion of mis-overlay: once a
    container at height ``h`` is mis-overlaid (its priority is larger than
    the minimum priority strictly below it), everything stacked above it is
    also considered mis-overlaid, since it must be moved before the
    offending container underneath can be accessed and re-sorted.

    Returns ``len(arr)`` when the stack has no mis-overlays (is "clean").
    """
    if len(arr) < 2:
        return len(arr)
    min_below = arr[0]
    for h in range(1, len(arr)):
        if arr[h] > min_below:
            return h
        if arr[h] < min_below:
            min_below = arr[h]
    return len(arr)


def is_clean_stack(arr: List[int]) -> bool:
    return first_bad_tier(arr) == len(arr)


def mis_overlay_count(stacks: Stacks) -> int:
    """Total number of mis-overlaid containers over the whole bay."""
    return sum(len(arr) - first_bad_tier(arr) for arr in stacks.values())


def is_sorted(stacks: Stacks) -> bool:
    """True iff the bay has zero mis-overlays (Section 2 of the paper)."""
    return mis_overlay_count(stacks) == 0


def num_empty_stacks(stacks: Stacks) -> int:
    return sum(1 for arr in stacks.values() if not arr)


def legal_moves(stacks: Stacks, max_tiers: int) -> List[Move]:
    """All applicable (src, dst) moves, ignoring branching/symmetry rules."""
    out: List[Move] = []
    keys = list(stacks.keys())
    for src in keys:
        if not stacks[src]:
            continue
        for dst in keys:
            if src == dst:
                continue
            if len(stacks[dst]) < max_tiers:
                out.append((src, dst))
    return out


def apply_move_sequence(stacks_init: Stacks, moves: List[Move], max_tiers: int) -> Stacks:
    """Return a *new* Stacks dict after applying ``moves`` in order."""
    st = clone_stacks(stacks_init)
    for mv in moves:
        apply_move_inplace(st, mv, max_tiers)
    return st
