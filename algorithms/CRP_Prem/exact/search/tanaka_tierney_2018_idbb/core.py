"""
Vendored IDBB wrappers for TanakaTierney2018IDBB.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, List, Tuple

Stacks = Dict[int, List[int]]
Move = Tuple[int, int]


def clone_stacks(stacks: Stacks) -> Stacks:
    return {k: list(v) for k, v in stacks.items()}


def is_well_located_stack(arr: List[int]) -> bool:
    return all(arr[i] >= arr[i + 1] for i in range(len(arr) - 1))


def is_fully_well_located(stacks: Stacks) -> bool:
    return all(is_well_located_stack(arr) for arr in stacks.values())


def count_bad_overlaps(stacks: Stacks) -> int:
    bad = 0
    for arr in stacks.values():
        for i in range(len(arr) - 1):
            if arr[i] < arr[i + 1]:
                bad += 1
    return bad


def apply_moves(stacks_init: Stacks, moves: List[Move], max_tiers: int) -> Stacks:
    st = clone_stacks(stacks_init)
    for src, dst in moves:
        if src == dst or src not in st or dst not in st:
            continue
        if not st[src] or len(st[dst]) >= max_tiers:
            continue
        st[dst].append(st[src].pop())
    return st
