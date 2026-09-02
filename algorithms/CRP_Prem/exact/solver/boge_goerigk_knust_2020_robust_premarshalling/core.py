"""
Uncertainty-set helpers for BogeGoerigkKnust2020RobustPMP.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

Stacks = Dict[int, List[int]]  # stack_idx -> class-ranks bottom..top


def clone_stacks(stacks: Stacks) -> Stacks:
    return {k: list(v) for k, v in stacks.items()}


def normalize_stacks_to_class_ranks(
    stacks_raw: Dict[int, List[int]],
) -> Tuple[Stacks, Dict[int, int], Dict[int, int]]:
    """
    Normalize arbitrary class labels to contiguous ranks 1..N.
    Returns:
      - normalized stacks
      - raw_class -> rank
      - rank -> raw_class
    """
    classes = sorted({x for arr in stacks_raw.values() for x in arr})
    raw_to_rank = {c: i + 1 for i, c in enumerate(classes)}
    rank_to_raw = {i + 1: c for i, c in enumerate(classes)}
    out: Stacks = {}
    for s, arr in stacks_raw.items():
        out[s] = [raw_to_rank[x] for x in arr]
    return out, raw_to_rank, rank_to_raw


def robust_upper_badly_placed(stacks: Stacks, delta: int) -> int:
    """
    Paper-style upper bound objective BIbar^rob:
    count items that may become badly placed in at least one scenario in U_delta.
    """
    bad_items: Set[Tuple[int, int]] = set()  # (stack, height)
    for s, arr in stacks.items():
        for h in range(1, len(arr)):
            above = arr[h]
            for j in range(h):
                below = arr[j]
                if below == above:
                    continue
                if abs(below - above) <= delta:
                    bad_items.add((s, h))
                    break
    return len(bad_items)


def count_badly_placed_under_rank(stacks: Stacks, rank_pos: Dict[int, int]) -> int:
    """
    Count badly placed items BI for one realized rank order of classes.
    """
    bad = 0
    for arr in stacks.values():
        for h in range(1, len(arr)):
            above = arr[h]
            is_bad = False
            for j in range(h):
                below = arr[j]
                if rank_pos[above] > rank_pos[below]:
                    is_bad = True
                    break
            if is_bad:
                bad += 1
    return bad


def enumerate_adjacent_swap_scenarios(
    n_classes: int,
    delta: int,
    max_scenarios: int,
) -> Optional[List[Tuple[int, ...]]]:
    """
    Enumerate all class-order scenarios within Kendall distance <= delta.
    Returns None when max_scenarios would be exceeded.
    """
    start = tuple(range(1, n_classes + 1))
    seen: Set[Tuple[int, ...]] = {start}
    dq = deque([(start, 0)])
    out: List[Tuple[int, ...]] = [start]

    while dq:
        perm, dist = dq.popleft()
        if dist >= delta:
            continue
        for i in range(n_classes - 1):
            nxt = list(perm)
            nxt[i], nxt[i + 1] = nxt[i + 1], nxt[i]
            nxt_t = tuple(nxt)
            if nxt_t in seen:
                continue
            seen.add(nxt_t)
            out.append(nxt_t)
            if len(out) > max_scenarios:
                return None
            dq.append((nxt_t, dist + 1))
    return out


def robust_exact_badly_placed(
    stacks: Stacks,
    n_classes: int,
    delta: int,
    max_scenarios: int,
) -> Optional[int]:
    """
    Exact BI^rob by explicit scenario enumeration for small class counts.
    Returns None when the uncertainty set is too large for the cap.
    """
    scenarios = enumerate_adjacent_swap_scenarios(
        n_classes=n_classes,
        delta=delta,
        max_scenarios=max_scenarios,
    )
    if scenarios is None:
        return None
    worst = 0
    for perm in scenarios:
        rank_pos = {cls: idx for idx, cls in enumerate(perm, start=1)}
        worst = max(worst, count_badly_placed_under_rank(stacks, rank_pos))
    return worst


def theorem3_has_robust_solution(
    n_items: int,
    n_stacks: int,
    max_tiers: int,
    delta: int,
) -> bool:
    free_slots = n_stacks * max_tiers - n_items
    return delta >= 1 and free_slots >= (max_tiers - 1) * delta


def theorem4_unique_priorities_has_robust_solution(
    n_items: int,
    n_stacks: int,
    max_tiers: int,
    delta: int,
) -> bool:
    """
    Unique-priority existence test (paper Theorem 4).
    """
    n = n_items
    m = n_stacks
    b = max_tiers
    d = delta
    if n <= d + 1:
        return n <= m
    if n <= (d + 1) * b:
        return (d + 1) <= m
    # case 3 in theorem: always feasible as long as storage can hold all items
    return (n + b - 1) // b <= m


def construct_chain_robust_layout(
    item_classes: Sequence[int],
    n_stacks: int,
    max_tiers: int,
    delta: int,
) -> Stacks:
    """
    Construct a delta-robust layout using class chains modulo (delta+1).
    Returns stack arrays bottom..top.
    """
    buckets: Dict[int, List[int]] = {}
    mod = max(1, delta + 1)
    for cls in item_classes:
        key = cls % mod
        buckets.setdefault(key, []).append(cls)
    for key in buckets:
        buckets[key].sort(reverse=True)  # bottom (later) -> top (earlier)

    pieces: List[List[int]] = []
    for key in sorted(buckets.keys()):
        arr = buckets[key]
        for i in range(0, len(arr), max_tiers):
            pieces.append(arr[i : i + max_tiers])

    out: Stacks = {s: [] for s in range(n_stacks)}
    for s in range(min(n_stacks, len(pieces))):
        out[s] = list(pieces[s])
    return out
