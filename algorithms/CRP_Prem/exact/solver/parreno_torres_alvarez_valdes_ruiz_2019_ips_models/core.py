"""
IPS6 shared helpers for ParrenoTorresAlvarezValdesRuiz2019IPS6.

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


def has_mis_overlay(arr: List[int]) -> bool:
    return any(arr[i] < arr[i + 1] for i in range(len(arr) - 1))


def is_well_located_stack(arr: List[int]) -> bool:
    return not has_mis_overlay(arr)


def is_fully_well_located(stacks: Stacks) -> bool:
    return all(is_well_located_stack(arr) for arr in stacks.values())


def count_bad_overlaps(stacks: Stacks) -> int:
    bad = 0
    for arr in stacks.values():
        for i in range(len(arr) - 1):
            if arr[i] < arr[i + 1]:
                bad += 1
    return bad


def distinct_priorities(stacks: Stacks) -> List[int]:
    return sorted({v for arr in stacks.values() for v in arr})


def coarsen_priorities(stacks: Stacks, max_priorities: int) -> Tuple[Stacks, Dict[int, int]]:
    """Bucket priorities into ``max_priorities`` contiguous groups when an
    instance has more distinct values than that, to keep the model
    tractable (documented approximation, not part of the original paper,
    which treats P as already given)."""
    values = distinct_priorities(stacks)
    if len(values) <= max_priorities:
        mapping = {v: v for v in values}
        return clone_stacks(stacks), mapping

    n = len(values)
    mapping = {}
    for i, v in enumerate(values):
        bucket = (i * max_priorities) // n
        mapping[v] = bucket + 1

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


def blocking_containers_lower_bound(stacks: Stacks) -> int:
    """
    Section 6's lower bound LB used to seed the iterative T-search: the
    number of "blocking containers" (Section 3's own definition -- a
    container c' above a container c with a worse, i.e. numerically larger,
    priority, together with everything stacked above c', all need at least
    one relocation).

    Per stack (bottom-to-top array ``arr``), the blocking set is the
    upward-closed tail starting at the lowest position where the running
    minimum priority seen below is violated -- equivalently, the first
    position (scanning bottom-up) whose value exceeds the minimum of
    everything below it. Every position from there to the top is blocking.
    A well-located stack contributes 0.
    """
    lb = 0
    for arr in stacks.values():
        n = len(arr)
        if n == 0:
            continue
        running_min = arr[0]
        first_violation = n  # no violation found => 0 blocking containers
        for j in range(1, n):
            if arr[j] > running_min:
                first_violation = j
                break
            running_min = min(running_min, arr[j])
        lb += n - first_violation
    return lb
