"""
Network-flow helpers for LeeHsu2007NetworkFlowIP.

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
    for i in range(1, len(arr)):
        if arr[i] > arr[i - 1]:
            return False
    return True


def is_fully_well_located(stacks: Stacks) -> bool:
    return all(is_well_located_stack(arr) for arr in stacks.values())


def count_bad_overlaps(stacks: Stacks) -> int:
    bad = 0
    for arr in stacks.values():
        for i in range(1, len(arr)):
            if arr[i] > arr[i - 1]:
                bad += 1
    return bad


def distinct_types(stacks: Stacks) -> List[int]:
    """Sorted list of distinct priority values present (ascending == most
    urgent first), used as the network's commodity/"type" set C."""
    return sorted({v for arr in stacks.values() for v in arr})


def coarsen_types(stacks: Stacks, max_types: int) -> Tuple[Stacks, Dict[int, int]]:
    """
    The paper explicitly treats C ("the number of container types") as a
    user-chosen parameter that trades model size for fidelity ("this
    parameter has a significant influence on the problem size"). When an
    instance has more distinct priority values than ``max_types``, bucket
    them into ``max_types`` contiguous groups (preserving relative order)
    so the network-flow model stays tractable. Containers sharing a bucket
    become mutually interchangeable for well-located purposes (a
    documented approximation, not part of the original paper).
    """
    values = distinct_types(stacks)
    if len(values) <= max_types:
        mapping = {v: v for v in values}
        return clone_stacks(stacks), mapping

    n = len(values)
    mapping = {}
    for i, v in enumerate(values):
        bucket = (i * max_types) // n
        mapping[v] = bucket + 1  # 1-indexed bucket "type value"

    coarse = {s: [mapping[v] for v in arr] for s, arr in stacks.items()}
    return coarse, mapping


def apply_moves(stacks_init: Stacks, moves: List[Move], max_tiers: int) -> Stacks:
    st = clone_stacks(stacks_init)
    for src, dst in moves:
        if src == dst or src not in st or dst not in st:
            continue
        if not st[src] or len(st[dst]) >= max_tiers:
            continue
        st[dst].append(st[src].pop())
    return st


def greedy_upper_bound_moves(stacks: Stacks, max_tiers: int) -> int:
    """
    Cheap, paper-independent greedy used *only* to pick a practical default
    number of time segments T for the network model -- not part of Lee &
    Hsu's method itself. Repeatedly relocates an out-of-place top container
    to the best available stack (one that keeps it well located there if
    possible, else the least-crowded stack).
    """
    st = clone_stacks(stacks)
    moves = 0
    n_containers = sum(len(v) for v in stacks.values())
    max_guard = 20 * (n_containers + 5)

    while not is_fully_well_located(st) and moves < max_guard:
        moved = False
        for s, arr in st.items():
            if len(arr) < 2 or arr[-1] <= arr[-2]:
                continue
            top = arr[-1]
            dest = None
            for z, arr2 in st.items():
                if z == s or len(arr2) >= max_tiers:
                    continue
                if not arr2 or arr2[-1] >= top:
                    dest = z
                    break
            if dest is None:
                candidates = [z for z in st if z != s and len(st[z]) < max_tiers]
                if not candidates:
                    continue
                dest = min(candidates, key=lambda z: len(st[z]))
            st[dest].append(st[s].pop())
            moves += 1
            moved = True
            break
        if not moved:
            break
    return moves
