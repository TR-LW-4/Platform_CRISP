"""
Greedy simulation and roulette-wheel helpers for CasertaLAH.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Primitive helpers
# ─────────────────────────────────────────────────────────────────────────────

def mins_value(prios: List[int], n_total: int) -> int:
    """mins(s) = minimum priority in stack s.  Empty stack → N+1."""
    return min(prios) if prios else n_total + 1


def _greedy_select(
    r:         int,
    src_key:   Any,
    stacks:    Dict[Any, List[int]],
    n_total:   int,
    max_tiers: int,
    all_keys:  List[Any],
) -> Optional[Any]:
    """
    Deterministic Min–Max destination selection.
    Returns the best destination key, or None if no valid stack.
    """
    best_good_key: Optional[Any] = None
    best_good_mins = float("inf")
    best_bad_key:  Optional[Any] = None
    best_bad_mins  = float("-inf")

    for key in all_keys:
        if key == src_key:
            continue
        if len(stacks[key]) >= max_tiers:
            continue
        ms = mins_value(stacks[key], n_total)
        if r < ms:                        # no new deadlock
            if ms < best_good_mins:
                best_good_mins = ms
                best_good_key  = key
        else:                             # new deadlock unavoidable
            if ms > best_bad_mins:
                best_bad_mins = ms
                best_bad_key  = key

    return best_good_key if best_good_key is not None else best_bad_key


def greedy_simulate(
    stacks_in: Dict[Any, List[int]],
    n_total:   int,
    max_tiers: int,
    all_keys:  List[Any],
) -> int:
    """
    Run the deterministic greedy heuristic to completion.
    Returns total relocations — used as the greedy score g(A').
    """
    total, _ = greedy_trajectory(stacks_in, n_total, max_tiers, all_keys)
    return total


def greedy_trajectory(
    stacks_in: Dict[Any, List[int]],
    n_total:   int,
    max_tiers: int,
    all_keys:  List[Any],
) -> Tuple[int, List[Any]]:
    """Return the greedy relocation count and its destination-stack sequence."""
    stacks = {k: list(v) for k, v in stacks_in.items()}
    total  = 0
    moves: List[Any] = []

    for target in range(1, n_total + 1):
        src_key = next((k for k, p in stacks.items() if target in p), None)
        if src_key is None:
            continue

        while stacks[src_key] and stacks[src_key][-1] != target:
            blocker = stacks[src_key][-1]
            dst = _greedy_select(blocker, src_key, stacks, n_total, max_tiers, all_keys)
            if dst is None:
                break
            stacks[src_key].pop()
            stacks[dst].append(blocker)
            moves.append(dst)
            total += 1

        if stacks[src_key] and stacks[src_key][-1] == target:
            stacks[src_key].pop()

    return total, moves


# ─────────────────────────────────────────────────────────────────────────────
# Look-ahead step: enumerate neighbours, score, roulette-wheel select
# ─────────────────────────────────────────────────────────────────────────────

def look_ahead_step(
    stacks:    Dict[Any, List[int]],
    n_total:   int,
    max_tiers: int,
    all_keys:  List[Any],
    rng:       np.random.RandomState,
) -> Optional[Any]:
    """
    Enumerate all feasible one-relocation neighbours of the current state,
    evaluate each with the greedy score g(A'), and return the chosen
    destination key via roulette-wheel selection.

    Returns None if no valid destination exists (pathological / full bay).
    """
    # Identify current target and its topmost blocker
    for target in range(1, n_total + 1):
        src_key = next((k for k, p in stacks.items() if target in p), None)
        if src_key is None:
            continue
        if stacks[src_key][-1] == target:
            continue       # target already on top; will be auto-retrieved
        blocker = stacks[src_key][-1]
        break
    else:
        return None   # bay empty or all targets accessible

    # Build candidate list and scores
    candidates: List[Any] = []
    scores:     List[int] = []

    for dst_key in all_keys:
        if dst_key == src_key:
            continue
        if len(stacks[dst_key]) >= max_tiers:
            continue

        # Simulate one relocation then run greedy to completion
        child = {k: list(v) for k, v in stacks.items()}
        child[src_key].pop()
        child[dst_key].append(blocker)
        score = greedy_simulate(child, n_total, max_tiers, all_keys)

        candidates.append(dst_key)
        scores.append(score)

    if not candidates:
        return None

    # Roulette-wheel: attractiveness = max_score − score + 1
    max_s = max(scores)
    attr  = np.array([max_s - s + 1 for s in scores], dtype=np.float64)
    probs = attr / attr.sum()
    idx   = rng.choice(len(candidates), p=probs)
    return candidates[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Single trajectory
# ─────────────────────────────────────────────────────────────────────────────

def run_trajectory(
    stacks_init: Dict[Any, List[int]],
    n_total:     int,
    max_tiers:   int,
    all_keys:    List[Any],
    rng:         np.random.RandomState,
    ub:          int,
) -> Tuple[int, List[Any]]:
    """
    Execute one complete trajectory using the look-ahead + roulette-wheel
    approach (Algorithm 1, lines 4–23).

    Parameters
    ----------
    ub  : current best upper bound — trajectory is abandoned early if
          relocations so far already reach or exceed ub.

    Returns
    -------
    Total relocations and destination-stack sequence.  An abandoned
    trajectory returns ``(ub+1, [])``.
    """
    stacks = {k: list(v) for k, v in stacks_init.items()}
    total  = 0
    moves: List[Any] = []

    for target in range(1, n_total + 1):
        src_key = next((k for k, p in stacks.items() if target in p), None)
        if src_key is None:
            continue

        while stacks[src_key] and stacks[src_key][-1] != target:
            # Trajectory fathoming: if already at/above best, abandon
            if total >= ub:
                return ub + 1, []

            dst = look_ahead_step(stacks, n_total, max_tiers, all_keys, rng)
            if dst is None:
                break

            blocker = stacks[src_key][-1]
            stacks[src_key].pop()
            stacks[dst].append(blocker)
            moves.append(dst)
            total += 1

        if stacks[src_key] and stacks[src_key][-1] == target:
            stacks[src_key].pop()

    return total, moves
