"""
Shared state helpers for the Tanaka & Tierney (2018) IDBB embedding.

Reference
---------
S. Tanaka, K. Tierney, "Solving real-world sized container pre-marshalling
problems with an iterative deepening branch-and-bound algorithm", European
Journal of Operational Research 264 (2018) 165-180.

This folder is self-contained on purpose: it does not import from (or get
imported by) any of the platform's other CRP-Prem embeddings, e.g.
``de_melo_silva_2018/`` or ``parreno_torres_alvarez_valdes_ruiz_2019_ips_models/``.
A small amount of duplicated boilerplate (stack helpers, identical to those
files) is the accepted trade-off, matching how every other paper folder
under ``exact/solver/`` in this platform is organised.

Convention (shared with this platform's other CRP-Prem embeddings):
stacks are lists of priority values, bottom to top; smaller values are
retrieved earlier and therefore belong closer to the top. A stack is well
located iff its values are non-increasing from bottom to top -- this
matches both ``core.yard.Stack.is_sorted_by_priority`` and the vendored
solver's own convention (verified empirically against the compiled
``pmp-1.02`` binary: it reads each ``Stack k: v0 v1 ...`` input line
left-to-right as bottom-to-top, and reports relocations in that same
orientation).
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
