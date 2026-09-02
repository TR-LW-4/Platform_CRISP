"""
Shared CPMPCT state helpers for ParrenoTorresAlvarezValdesRuizTierney2020CPMPCT.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, List, Tuple

Stacks = Dict[int, List[int]]  # stack_idx -> priorities bottom..top
Move = Tuple[int, int]         # (src, dst)


def clone_stacks(stacks: Stacks) -> Stacks:
    return {k: list(v) for k, v in stacks.items()}


def total_bad_overlaps(stacks: Stacks) -> int:
    bad = 0
    for arr in stacks.values():
        for h in range(1, len(arr)):
            x = arr[h]
            for j in range(h):
                if arr[j] < x:
                    bad += 1
                    break
    return bad


def is_sorted(stacks: Stacks) -> bool:
    return total_bad_overlaps(stacks) == 0


def top_priority(stacks: Stacks, s: int) -> int:
    arr = stacks.get(s, [])
    if not arr:
        return 10**9
    return int(arr[-1])


def legal_moves(stacks: Stacks, max_tiers: int) -> List[Move]:
    out: List[Move] = []
    keys = sorted(stacks.keys())
    for src in keys:
        if not stacks[src]:
            continue
        for dst in keys:
            if src == dst:
                continue
            if len(stacks[dst]) < max_tiers:
                out.append((src, dst))
    return out


def apply_move_inplace(stacks: Stacks, src: int, dst: int, max_tiers: int) -> bool:
    if src == dst:
        return False
    if src not in stacks or dst not in stacks:
        return False
    if not stacks[src] or len(stacks[dst]) >= max_tiers:
        return False
    x = stacks[src].pop()
    stacks[dst].append(x)
    return True


def blocking_tiers(stacks: Stacks) -> List[int]:
    """
    Return tiers (1-based from bottom) of initially blocking containers.
    """
    out: List[int] = []
    for arr in stacks.values():
        for h in range(1, len(arr)):
            x = arr[h]
            is_blocking = False
            for j in range(h):
                if arr[j] < x:
                    is_blocking = True
                    break
            if is_blocking:
                out.append(h + 1)
    return out
