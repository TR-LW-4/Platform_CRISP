"""
OPT(n) DP operator for FeillletLS.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .solution import Step, rebuild_solution


# ═══════════════════════════════════════════════════════════════════ #
#  Public entry point                                                 #
# ═══════════════════════════════════════════════════════════════════ #

def opt_n(
    n:           int,
    steps:       List[Step],
    init_stacks: List[Optional[List[int]]],  # 1-indexed; index 0 = None
    W:           int,
    H_max:       int,
    N:           int,
) -> Optional[List[Step]]:
    """
    Reoptimise container n's relocation sequence within *steps*.

    Returns a new step list (with n's moves replaced) if a strictly shorter
    sequence for n exists, otherwise returns None.
    """
    f_n = sum(1 for s in steps if s.container == n and s.dst != 0)
    if f_n == 0:
        return None  # already at lower bound (0 relocations)

    steps_mn, h, sigma, M, s0, h0 = _build_s_minus_n(steps, n, init_stacks, W)
    if M <= 1:
        return None

    max_h = _suffix_max(h, W, M)
    min_h = _suffix_min(h, W, M)

    INF = f_n          # upper bound: must strictly improve
    HH  = H_max + 2    # tier index sentinel

    # f_cur[s][h_tier]: min cost to have n at (s, h_tier) in current S^−n config
    f_cur: List[List[int]] = [[INF] * HH for _ in range(W + 1)]
    f_cur[s0][h0] = 0
    Q: List[Tuple[int, int]] = [(s0, h0)]

    # Parent pointers: all_par[t][(s, h_tier)] = (prev_s, prev_h, relocated)
    # relocated=True  →  n moved from (prev_s, ?) to (s, ?) before S^−n step t−1
    all_par: List[Optional[Dict[Tuple[int, int], Tuple[int, int, bool]]]] = [None] * (M + 1)
    all_par[1] = {(s0, h0): (s0, h0, False)}  # dummy root

    for t in range(1, M):
        f_nxt: List[List[int]] = [[INF] * HH for _ in range(W + 1)]
        par_t: Dict[Tuple[int, int], Tuple[int, int, bool]] = {}
        Q_next: set = set()

        for s, h_tier in Q:
            cost = f_cur[s][h_tier]
            if cost >= INF:
                continue

            # ── Cost-0 transition: n stays at (s, h_tier) ─────────────────── #
            if _feasible(t + 1, s, h_tier, h, sigma, H_max, M):
                if cost < f_nxt[s][h_tier]:
                    f_nxt[s][h_tier] = cost
                    par_t[(s, h_tier)] = (s, h_tier, False)
                    Q_next.add((s, h_tier))
                    if cost < INF and _aspiration(t + 1, s, h_tier, h, max_h, min_h, M, H_max):
                        all_par[t + 1] = par_t
                        return _finish(
                            all_par, t + 1, s, h_tier, n, steps, steps_mn, s0, h0, M
                        )

            # ── Cost-1 transition: relocate n (only if n is on top) ────────── #
            if h_tier != h[s][t] + 1:
                continue
            new_cost = cost + 1
            if new_cost >= INF:           # Speedup #1: prune by upper bound
                continue

            for s2 in range(1, W + 1):
                if s2 == s:
                    continue
                h2 = h[s2][t + 1] + 1
                if h2 >= HH or h2 > H_max:
                    continue
                if _feasible(t + 1, s2, h2, h, sigma, H_max, M):
                    if new_cost < f_nxt[s2][h2]:
                        f_nxt[s2][h2] = new_cost
                        par_t[(s2, h2)] = (s, h_tier, True)
                        Q_next.add((s2, h2))
                        if new_cost < INF and _aspiration(
                            t + 1, s2, h2, h, max_h, min_h, M, H_max
                        ):
                            all_par[t + 1] = par_t
                            return _finish(
                                all_par, t + 1, s2, h2, n, steps, steps_mn, s0, h0, M
                            )

        f_cur      = f_nxt
        all_par[t + 1] = par_t
        Q          = list(Q_next)
        if not Q:
            break

    # ── Check final configurations at t = M ───────────────────────────── #
    best_cost = INF
    best_s = best_h = -1
    for s in range(1, W + 1):
        h_final = h[s][M] + 1
        if h_final >= HH or h_final > H_max:
            continue
        if f_cur[s][h_final] < best_cost:
            best_cost      = f_cur[s][h_final]
            best_s, best_h = s, h_final

    if best_cost >= f_n:
        return None

    return _finish(all_par, M, best_s, best_h, n, steps, steps_mn, s0, h0, M)


# ═══════════════════════════════════════════════════════════════════ #
#  Private helpers                                                    #
# ═══════════════════════════════════════════════════════════════════ #

def _build_s_minus_n(
    steps:       List[Step],
    n:           int,
    init_stacks: List[Optional[List[int]]],
    W:           int,
) -> Tuple[List[Step], List[List[int]], List[Tuple[int, int]], int, int, int]:
    """
    Build S^−n and pre-compute the height matrix.

    Returns
    -------
    steps_mn  S^−n steps (before n's retrieval, excluding n's relocations)
    h         h[s][t] for s=1..W, t=1..M  (stack heights in S^−n)
    sigma     sigma[k] = (src, dst) for k=0..M-2  (S^−n step k+1)
    M         number of S^−n configurations
    s0, h0    initial stack and tier of n (1-indexed)
    """
    s0 = h0 = 0
    for s in range(1, W + 1):
        stk = init_stacks[s]
        if stk and n in stk:
            s0 = s
            h0 = stk.index(n) + 1  # 1-indexed tier from bottom
            break

    steps_mn: List[Step] = []
    for step in steps:
        if step.container == n and step.dst == 0:
            break
        if step.container != n:
            steps_mn.append(step)

    M = len(steps_mn) + 1

    # Bay copy with n removed
    curr: List[Optional[List[int]]] = [None] + [list(init_stacks[s]) for s in range(1, W + 1)]  # type: ignore[index]
    if s0 > 0:
        curr[s0].remove(n)  # type: ignore[union-attr]

    h: List[List[int]] = [[0] * (M + 1) for _ in range(W + 1)]
    for s in range(1, W + 1):
        h[s][1] = len(curr[s])  # type: ignore[arg-type]

    sigma: List[Tuple[int, int]] = []
    for k, step in enumerate(steps_mn):
        src, dst = step.src, step.dst
        sigma.append((src, dst))
        curr[src].pop()  # type: ignore[union-attr]
        if dst != 0:
            curr[dst].append(step.container)  # type: ignore[union-attr]
        for s in range(1, W + 1):
            h[s][k + 2] = len(curr[s])  # type: ignore[arg-type]

    return steps_mn, h, sigma, M, s0, h0


def _feasible(
    t:     int,
    s:     int,
    h_tier: int,
    h:     List[List[int]],
    sigma: List[Tuple[int, int]],
    H_max: int,
    M:     int,
) -> bool:
    """
    Check whether STATE(t, s, h_tier) is feasible (Definition 1).

    Uses rules R1–R4 for middle states and F1–F2 for the final state.
    """
    ht = h[s][t]
    if t == M:
        return h_tier == ht + 1 and ht < H_max

    # Middle state: 2 ≤ t ≤ M−1
    if h_tier > ht + 1:                      # R1: not floating
        return False
    if ht >= H_max:                           # R2: room for n
        return False
    sig = sigma[t - 1]                        # step σ^t (from config t to t+1)
    if sig[0] == s and h_tier > ht:           # R3: n not on top when crane picks from s
        return False
    if sig[1] == s and ht + 1 >= H_max:      # R4: receiving won't overflow
        return False
    return True


def _suffix_max(h: List[List[int]], W: int, M: int) -> List[List[int]]:
    """max_h[s][t] = max h(s, t') for t' = t..M"""
    mx: List[List[int]] = [[0] * (M + 2) for _ in range(W + 1)]
    for s in range(1, W + 1):
        mx[s][M] = h[s][M]
        for t in range(M - 1, 0, -1):
            mx[s][t] = max(h[s][t], mx[s][t + 1])
    return mx


def _suffix_min(h: List[List[int]], W: int, M: int) -> List[List[int]]:
    """min_h[s][t] = min h(s, t') for t' = t..M"""
    mn: List[List[int]] = [[0] * (M + 2) for _ in range(W + 1)]
    for s in range(1, W + 1):
        mn[s][M] = h[s][M]
        for t in range(M - 1, 0, -1):
            mn[s][t] = min(h[s][t], mn[s][t + 1])
    return mn


def _aspiration(
    t:     int,
    s:     int,
    h_tier: int,
    h:     List[List[int]],
    max_h: List[List[int]],
    min_h: List[List[int]],
    M:     int,
    H_max: int,
) -> bool:
    """
    Aspiration criterion (Speedup #3, Section 3.2.3 of Feillet et al.).

    Returns True if n can remain at (s, h_tier) from config t to M without
    any further relocations and end up exactly on top for retrieval.

    Conditions:
      A1. max h(s, t') < H_max  for t' ≥ t  →  always room for n
      A2. min h(s, t') ≥ h_tier−1  for t' ≥ t  →  floor under n never removed
      A3. h(s, M) = h_tier−1  →  n is exactly on top at the final config
    """
    return (
        max_h[s][t] < H_max
        and min_h[s][t] >= h_tier - 1
        and h[s][M] == h_tier - 1
    )


def _finish(
    all_par:  List[Optional[Dict]],
    t_cur:    int,
    s_cur:    int,
    h_cur:    int,
    n:        int,
    steps:    List[Step],
    steps_mn: List[Step],
    s0:       int,
    h0:       int,
    M:        int,
) -> List[Step]:
    """
    Backtrack parent pointers from (t_cur, s_cur, h_cur) to the initial state
    (1, s0, h0), collecting relocation events in forward order, then rebuild
    the full solution.
    """
    events: List[Tuple[int, int, int]] = []  # (t_before, src, dst)

    s, h_tier, t = s_cur, h_cur, t_cur
    while t > 1:
        par_at_t = all_par[t]
        if par_at_t is None:
            break
        entry = par_at_t.get((s, h_tier))
        if entry is None:
            break
        prev_s, prev_h, relocated = entry
        if relocated:
            # n was relocated before S^−n step t−1 (1-indexed), from prev_s to s
            events.append((t - 1, prev_s, s))
        s, h_tier = prev_s, prev_h
        t -= 1

    events.reverse()
    return rebuild_solution(steps, n, events, s_cur, steps_mn)
