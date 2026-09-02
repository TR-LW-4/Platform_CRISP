"""
Proposition-1 lower bound for BortfeldtForster2012TreeSearch.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List, Tuple

from .core import (
    Stacks,
    bad_placed_count,
    bad_placed_per_stack,
    cumulative_demand,
    cumulative_potential_supply,
    first_bad_tier,
    highest_well_placed_group,
    is_clean_stack,
)


def lb_bg_only(stacks: Stacks) -> int:
    """Naive baseline n0_BG = nb (used for the paper §7.5 V1 ablation)."""
    return bad_placed_count(stacks)


def lb_bx(stacks: Stacks) -> int:
    """Proposition 1 (i): n0_BX = nb + min{ nb(s) }."""
    nb = bad_placed_count(stacks)
    if nb == 0:
        return 0
    if any(is_clean_stack(stacks, s) for s in stacks):
        return nb
    per = bad_placed_per_stack(stacks)
    if not per:
        return nb
    return nb + min(per.values())


def _argmax_ds(stacks: Stacks, num_groups: int, max_tiers: int) -> Tuple[int, int]:
    """
    Return (Ds(g*), g*) where g* maximizes Ds(g) = D(g) - Sp(g).
    Ties broken by preferring smaller g (higher priority group), which favours
    early demand and matches the paper's ordering intuition.
    """
    best_ds = 0
    best_g = 1
    first = True
    for g in range(1, num_groups + 1):
        D = cumulative_demand(stacks, g, num_groups)
        Sp = cumulative_potential_supply(stacks, g, num_groups, max_tiers)
        ds = D - Sp
        if first or ds > best_ds or (ds == best_ds and g < best_g):
            best_ds = ds
            best_g = g
            first = False
    return best_ds, best_g


def _ng_star_of_stack(stacks: Stacks, s: int, g_star: int) -> int:
    """Number of well-placed items in stack s with group < g*."""
    arr = stacks[s]
    hb = first_bad_tier(arr)
    return sum(1 for h in range(hb) if arr[h] < g_star)


def lb_gx(stacks: Stacks, num_groups: int, max_tiers: int) -> int:
    """
    Proposition 1 (ii): lower bound on the number of GX moves.

    Returns 0 whenever Ds(g*) <= 0 (no residual demand surplus).
    """
    ds_star, g_star = _argmax_ds(stacks, num_groups, max_tiers)
    if ds_star <= 0:
        return 0
    ns_gx = -(-int(ds_star) // int(max_tiers))  # ceil division
    if ns_gx <= 0:
        return 0
    pot: List[int] = []
    for s, arr in stacks.items():
        if not arr:
            continue
        hw = highest_well_placed_group(stacks, s)
        if hw is None or hw >= g_star:
            continue
        pot.append(_ng_star_of_stack(stacks, s, g_star))
    if not pot:
        return 0
    pot.sort()
    take = min(ns_gx, len(pot))
    return sum(pot[:take])


def lb_moves(stacks: Stacks, num_groups: int, max_tiers: int) -> int:
    """Proposition 1 (iii): n0_m = n0_BX + n0_GX."""
    return lb_bx(stacks) + lb_gx(stacks, num_groups, max_tiers)
