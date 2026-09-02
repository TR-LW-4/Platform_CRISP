"""
Shared stage helpers for JovanovicTubaVoss2015MultiHeuristic.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

Stacks = Dict[int, List[int]]
Move = Tuple[int, int]
ContainerRef = Tuple[int, int]  # (stack, tier index from bottom, 0-based)


def clone_stacks(stacks: Stacks) -> Stacks:
    return {k: list(v) for k, v in stacks.items()}


def well_located_height(arr: Sequence[int]) -> int:
    """Length of the maximal non-increasing prefix counted from the bottom."""
    if len(arr) < 2:
        return len(arr)
    min_below = arr[0]
    for i in range(1, len(arr)):
        if arr[i] > min_below:
            return i
        if arr[i] < min_below:
            min_below = arr[i]
    return len(arr)


def is_well_located_stack(arr: Sequence[int]) -> bool:
    return well_located_height(arr) == len(arr)


def is_well_located(stacks: Stacks, ref: ContainerRef) -> bool:
    s, t = ref
    return t < well_located_height(stacks[s])


def not_well_located_refs(stacks: Stacks) -> List[ContainerRef]:
    out: List[ContainerRef] = []
    for s, arr in stacks.items():
        wh = well_located_height(arr)
        for t in range(wh, len(arr)):
            out.append((s, t))
    return out


def not_well_located_with_priority(stacks: Stacks, p: int) -> List[ContainerRef]:
    out: List[ContainerRef] = []
    for s, arr in stacks.items():
        wh = well_located_height(arr)
        for t in range(wh, len(arr)):
            if arr[t] == p:
                out.append((s, t))
    return out


def g_count(stacks: Stacks, ref: ContainerRef) -> int:
    """g(c, s): number of blocks above c in its own (source) stack."""
    s, t = ref
    return len(stacks[s]) - t - 1


def _landing_boundary(arr: Sequence[int], p: int) -> int:
    """
    Highest index (0-indexed) whose value is >= p; -1 if none (or if arr is
    empty). This is the container c := p could legally rest directly on
    top of (equal values may be stacked); everything above this index has
    a strictly smaller value than p and must be cleared first.
    """
    last = -1
    for t, v in enumerate(arr):
        if v >= p:
            last = t
    return last


def f_count(stacks: Stacks, s_star: int, p: int) -> int:
    """f(c, s*): number of blocks to remove from s* to well-locate c := p there."""
    arr = stacks[s_star]
    last = _landing_boundary(arr, p)
    return len(arr) - (last + 1)


def nw_count(stacks: Stacks, s_star: int, p: int) -> int:
    """nw(c, s*): how many of the f(c, s*) cleared blocks are well located."""
    arr = stacks[s_star]
    if not arr:
        return 0
    last = _landing_boundary(arr, p)
    boundary = last + 1
    wh = well_located_height(arr)
    return max(0, wh - boundary)


def w_score(stacks: Stacks, ref: ContainerRef, s_star: int) -> int:
    """w(c, s*), Eq. 1."""
    s, _ = ref
    p = stacks[s][ref[1]]
    f = f_count(stacks, s_star, p)
    if s == s_star:
        return f + 1
    return f + g_count(stacks, ref) + 1


def w_hat_score(stacks: Stacks, ref: ContainerRef, s_star: int) -> int:
    """w-hat(c, s*), Eq. 2: replaces f by f-hat = f + nw."""
    s, t = ref
    p = stacks[s][t]
    f_hat = f_count(stacks, s_star, p) + nw_count(stacks, s_star, p)
    if s == s_star:
        return f_hat + 1
    return f_hat + g_count(stacks, ref) + 1


def best_destination(
    stacks: Stacks,
    ref: ContainerRef,
    use_improved: bool,
    max_tiers: int,
) -> Optional[int]:
    """
    d(c) / Eq. 3: argmin over stacks with room of w (or w-hat if
    ``use_improved``); ties broken by the lower stack index.
    """
    score_fn = w_hat_score if use_improved else w_score
    best_s: Optional[int] = None
    best_score: Optional[int] = None
    for s in sorted(stacks.keys()):
        if len(stacks[s]) >= max_tiers:
            continue
        sc = score_fn(stacks, ref, s)
        if best_score is None or sc < best_score:
            best_score = sc
            best_s = s
    return best_s


def _effective_top(arr: Sequence[int]) -> int:
    """Per Sec. 3.2: well located (or empty) stacks are treated as top = 0."""
    if not arr or is_well_located_stack(arr):
        return 0
    return arr[-1]


def blocking_values(stacks: Stacks, ref: ContainerRef, s_star: int) -> List[int]:
    """
    Priority values of every block that stage 3 would need to temporarily
    relocate to well-locate the container at ``ref`` onto ``s_star``: the
    g(c, s) blocks above it in its own stack, plus the f(c, s*) blocks
    currently blocking the landing spot in s* (only when s != s*).
    """
    s, t = ref
    p = stacks[s][t]
    values = list(stacks[s][t + 1:])
    if s != s_star:
        f = f_count(stacks, s_star, p)
        if f > 0:
            values.extend(stacks[s_star][-f:])
    return values


def forced_relocations(stacks: Stacks, ref: ContainerRef, s_star: int) -> int:
    """fr(c, s*): approximate forced-relocation count, see module docstring."""
    s, _ = ref
    others = [_effective_top(stacks[k]) for k in stacks if k not in (s, s_star)]
    if not others:
        return 0
    min_top = min(others)
    return sum(1 for v in blocking_values(stacks, ref, s_star) if v < min_top)


def h_hat_score(stacks: Stacks, ref: ContainerRef, max_tiers: int) -> Optional[Tuple[int, int]]:
    """
    h-hat(c), Eq. 4: -p(c) + w-hat(c, d(c)) + fr(c, d(c)). Returns None if no
    destination stack with room exists (should not normally happen since c's
    own stack always has room for itself when s == d(c) is disallowed... but
    guarded defensively).
    """
    s, t = ref
    p = stacks[s][t]
    d = best_destination(stacks, ref, use_improved=True, max_tiers=max_tiers)
    if d is None:
        return None
    w_hat = w_hat_score(stacks, ref, d)
    fr = forced_relocations(stacks, ref, d)
    return (-p + w_hat + fr, d)


def select_next_block(
    stacks: Stacks,
    use_lookahead: bool,
    max_tiers: int,
) -> Optional[Tuple[ContainerRef, int]]:
    """
    Stage 2 (Sec. 3.2): pick the next container to well-locate.

    ``use_lookahead=False`` reproduces the original algorithm's strict
    descending-due-date order (the highest not-yet-resolved priority value).
    ``use_lookahead=True`` is the h-hat-based selection (Eq. 3-5), evaluated
    over *all* not-well-located containers.

    Returns ``(ref, priority)`` of the chosen container, or ``None`` if the
    layout is already fully well located.
    """
    refs = not_well_located_refs(stacks)
    if not refs:
        return None

    if not use_lookahead:
        p_max = max(stacks[s][t] for s, t in refs)
        candidates = [r for r in refs if stacks[r[0]][r[1]] == p_max]
        chosen = min(candidates)  # lowest (stack, tier) index for determinism
        return chosen, p_max

    best_ref = None
    best_key = None
    for ref in refs:
        scored = h_hat_score(stacks, ref, max_tiers)
        if scored is None:
            continue
        key, _ = scored
        if best_key is None or key < best_key:
            best_key = key
            best_ref = ref
    if best_ref is None:
        # Fall back to plain descending order if h-hat is degenerate
        # (e.g. every stack is full except the container's own).
        p_max = max(stacks[s][t] for s, t in refs)
        candidates = [r for r in refs if stacks[r[0]][r[1]] == p_max]
        best_ref = min(candidates)
    return best_ref, stacks[best_ref[0]][best_ref[1]]
