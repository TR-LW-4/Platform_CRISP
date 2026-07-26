"""
Cost estimation heuristics ``h(n)`` for A*/IDA* (Section 4.3 of the paper).

Two lower bounds on the number of moves needed to reach a mis-overlay-free
layout are provided:

- ``lb_direct``: the "direct" lower bound used by the state-of-the-art A*
  baseline of Expósito-Izquierdo et al. (2012) -- simply the number of
  mis-overlaid containers currently in the bay.

- ``lb_emo``: the "extended mis-overlay" (EMO) bound, a re-derivation of the
  Bortfeldt & Forster (2012) supply/demand lower bound (their Proposition 1),
  which the paper adopts as its main A*/IDA* heuristic since it dominates
  ``lb_direct`` (Section 4.3). Priority values double as "container groups"
  in the Bortfeldt & Forster sense: a smaller value must leave sooner and
  therefore belongs closer to the top of a mis-overlay-free stack.

Both bounds are *admissible* (never overestimate the true number of moves
needed). The EMO bound is admissible but *not consistent* -- as noted in
Section 4.3, applying a move can, in rare cases, cause it to increase. A*/IDA*
require a consistent heuristic, so callers should clamp a child's bound to be
no lower than its parent's (handled by :func:`emo_bound_consistent` /
the search module, not here).

This module is self-contained and does not import from any sibling
algorithm package (e.g. ``heuristic/bortfeldt_forster_2012_tree_search``),
even though it independently re-implements the same published formula.
"""

from __future__ import annotations

from typing import Dict, List

from .core import Stacks, first_bad_tier, mis_overlay_count, num_empty_stacks


def lb_direct(stacks: Stacks) -> int:
    """Direct lower bound: count of mis-overlaid containers."""
    return mis_overlay_count(stacks)


def _highest_well_placed_priority(arr: List[int]) -> int:
    """Priority of the highest well-placed (non mis-overlaid) container.

    Returns 0 (sentinel, smaller than any real priority) for an empty or
    fully mis-overlaid-from-the-bottom stack.
    """
    hb = first_bad_tier(arr)
    if hb == 0:
        return 0
    return arr[hb - 1]


def _bad_placed_per_stack(stacks: Stacks) -> Dict[int, int]:
    return {s: len(arr) - first_bad_tier(arr) for s, arr in stacks.items()}


def _max_priority(stacks: Stacks) -> int:
    best = 0
    for arr in stacks.values():
        for p in arr:
            if p > best:
                best = p
    return best


def _potential_supply_slots(stacks: Stacks, g: int, max_tiers: int) -> int:
    """sp(g): free slots above a stack whose highest well-placed item is g."""
    total = 0
    for arr in stacks.values():
        if not arr:
            continue
        hb = first_bad_tier(arr)
        if hb == 0:
            continue
        if arr[hb - 1] != g:
            continue
        total += max(0, max_tiers - hb)
    return total


def _cumulative_potential_supply(stacks: Stacks, g: int, num_groups: int, max_tiers: int) -> int:
    """Sp(g) = sp(g) + ... + sp(G) + H * (number of empty stacks)."""
    inner = sum(_potential_supply_slots(stacks, gg, max_tiers) for gg in range(g, num_groups + 1))
    return inner + max_tiers * num_empty_stacks(stacks)


def _demand(stacks: Stacks, g: int) -> int:
    """d(g): number of mis-overlaid containers with priority (group) g."""
    d = 0
    for arr in stacks.values():
        hb = first_bad_tier(arr)
        for h in range(hb, len(arr)):
            if arr[h] == g:
                d += 1
    return d


def _cumulative_demand(stacks: Stacks, g: int, num_groups: int) -> int:
    return sum(_demand(stacks, gg) for gg in range(g, num_groups + 1))


def _lb_bx(stacks: Stacks) -> int:
    """Proposition 1(i): n0_BX = nb + min_s{nb(s)} (0 if a clean stack exists)."""
    nb = mis_overlay_count(stacks)
    if nb == 0:
        return 0
    per_stack = _bad_placed_per_stack(stacks)
    if any(v == 0 for v in per_stack.values()):
        return nb
    if not per_stack:
        return nb
    return nb + min(per_stack.values())


def _lb_gx(stacks: Stacks, num_groups: int, max_tiers: int) -> int:
    """Proposition 1(ii): lower bound on the number of extra ("GX") moves."""
    best_ds = 0
    best_g = 1
    first = True
    for g in range(1, num_groups + 1):
        d = _cumulative_demand(stacks, g, num_groups)
        sp = _cumulative_potential_supply(stacks, g, num_groups, max_tiers)
        ds = d - sp
        if first or ds > best_ds or (ds == best_ds and g < best_g):
            best_ds, best_g = ds, g
            first = False
    if best_ds <= 0:
        return 0
    ns_gx = -(-int(best_ds) // int(max_tiers))  # ceil division
    if ns_gx <= 0:
        return 0

    pot: List[int] = []
    for s, arr in stacks.items():
        if not arr:
            continue
        hw = _highest_well_placed_priority(arr)
        if hw == 0 or hw >= best_g:
            continue
        hb = first_bad_tier(arr)
        pot.append(sum(1 for h in range(hb) if arr[h] < best_g))
    if not pot:
        return 0
    pot.sort()
    return sum(pot[: min(ns_gx, len(pot))])


def lb_emo(stacks: Stacks, max_tiers: int, num_groups: int = 0) -> int:
    """
    Extended mis-overlay (EMO) lower bound (Bortfeldt & Forster 2012,
    Proposition 1): n0_m = n0_BX + n0_GX. Dominates :func:`lb_direct`.
    """
    if not num_groups:
        num_groups = _max_priority(stacks)
    if num_groups <= 0:
        return 0
    return _lb_bx(stacks) + _lb_gx(stacks, num_groups, max_tiers)
