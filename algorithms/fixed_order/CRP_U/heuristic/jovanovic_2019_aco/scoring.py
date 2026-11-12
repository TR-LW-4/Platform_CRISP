"""
Gre-C candidate list and pheromone helpers for JovanovicACO_uBRP.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Scoring primitives
# ─────────────────────────────────────────────────────────────────────────────

def dif(c: int, d: int, n_total: int) -> int:
    """
    Eq. 3 — undesirability of placing container c above a stack whose
    minimum due-date is d.

    dif ≤ N  →  c will be well-located (d > c, d retrieved after c).
    dif > N  →  c will NOT be well-located (d ≤ c, d retrieved before c).
    Range: 1 .. 2N.  Smaller = more desirable destination.
    """
    if d > c:
        return d - c
    return 2 * n_total + 1 - d


def dife(c: int, d: int, imp_c: int, is_wl: bool, n_total: int) -> int:
    """
    Eq. 21 — extended dif for selecting among well-located and non-well-located
    candidate containers.

    For a non-well-located container: dife = dif(c, d, n_total).
    For a well-located container:     dife = N + dif(c, d, n_total) - imp(c).
      The N penalty accounts for the extra relocation that well-located c adds;
      imp(c) offsets it by the improvement gained in c's source stack.
    """
    base = dif(c, d, n_total)
    if is_wl:
        return n_total + base - imp_c
    return base


def dd_star(stack_priorities: List[int], n_total: int, stack_1based_idx: int) -> int:
    """
    dd*(S) — extended dd for pheromone-matrix indexing.

    Non-empty stack: returns min due-date ∈ {1 .. N}.
    Empty stack    : returns N + stack_1based_idx ∈ {N+1 .. N+W}.
      Differentiates empty stacks so the pheromone matrix can learn which
      empty slot is most beneficial.
    """
    if not stack_priorities:
        return n_total + stack_1based_idx
    return min(stack_priorities)


# ─────────────────────────────────────────────────────────────────────────────
# Lower bound
# ─────────────────────────────────────────────────────────────────────────────

def compute_lb(stacks: Dict[Any, List[int]]) -> int:
    """
    uBRP lower bound: total count of non-well-located containers.

    Container c at tier i (bottom = 0) is non-well-located if any container
    below it has a smaller due-date, i.e. min(prios[:i]) < c.
    """
    lb = 0
    for prios in stacks.values():
        for i in range(1, len(prios)):
            if min(prios[:i]) < prios[i]:
                lb += 1
    return lb


# ─────────────────────────────────────────────────────────────────────────────
# Candidate-list construction (Gre-C, Eqs. 9–23)
# ─────────────────────────────────────────────────────────────────────────────

def build_candidates_grec(
    stacks:    Dict[Any, List[int]],
    loc:       Dict[int, Any],
    smin:      Dict[Any, int],
    all_keys:  List[Any],
    src_key:   Any,
    target:    int,
    n_total:   int,
    max_tiers: int,
) -> List[Tuple[int, Any, Any]]:
    """
    Build the Gre-C candidate list Ĉ for one relocation step.

    Returns list of (container_due_date, source_key, destination_key).

    Candidate sets (paper notation):
      T   — topmost blocker above target → any valid destination (Eq. 11)
      Tn  — non-well-located tops NOT in src_key stack (Eq. 10)
      Or  — (c ∈ Tn) → destinations where c becomes well-located (Eq. 12)
      Cr  = T ∪ Or  (Eq. 13, used when NR ≠ ∅)
      Tw  — well-located tops whose removal enables Tn-container well-location
            (Eqs. 16–18, used when NR = ∅)
      Ow  — relocations of Tw containers (Eq. 19)
      Cw  = T ∪ Ow  (Eq. 20, used when NR = ∅)
      Ĉ   = Cr if NR ≠ ∅ else Cw  (Eq. 23)
    """
    # ── T: blocker of current target → any valid destination ────────── #
    T: List[Tuple[int, Any, Any]] = []
    if stacks[src_key] and stacks[src_key][-1] != target:
        top_c = stacks[src_key][-1]
        for dst in all_keys:
            if dst != src_key and len(stacks[dst]) < max_tiers:
                T.append((top_c, src_key, dst))

    # ── Tn: non-WL tops NOT in src_key (Eq. 10) ─────────────────────── #
    Tn: List[int] = []
    for key in all_keys:
        if key == src_key or not stacks[key]:
            continue
        c = stacks[key][-1]
        if smin[key] < c:       # something below c has smaller priority → non-WL
            Tn.append(c)

    # ── Or: Tn containers → destinations where they become well-located ─ #
    Or: List[Tuple[int, Any, Any]] = []
    for c in Tn:
        c_key = loc[c]
        for dst in all_keys:
            if dst == c_key or len(stacks[dst]) >= max_tiers:
                continue
            if smin[dst] > c:   # dd(dst) > c → c will be well-located there
                Or.append((c, c_key, dst))

    # ── NR: any non-WL container (incl. top blocker) that can be WL'd ── #
    NR_exists = len(Or) > 0
    if not NR_exists and T:
        top_c_val = T[0][0]
        for dst in all_keys:
            if dst != src_key and len(stacks[dst]) < max_tiers and smin[dst] > top_c_val:
                NR_exists = True
                break

    if NR_exists:
        # Cr = T ∪ Or
        return T + Or

    # ── NR = ∅: consider Ow (well-located tops that unlock future moves) #
    max_n = max(Tn) if Tn else 0
    Tn_set = set(Tn)

    # Tw: well-located tops (Eqs. 17–18): dd(S, t(c)−1) > max_n
    Ow: List[Tuple[int, Any, Any]] = []
    for key in all_keys:
        if key == src_key or not stacks[key]:
            continue
        c = stacks[key][-1]
        if c in Tn_set or c == target:
            continue
        if smin[key] < c:
            continue    # not well-located
        # dd(S, t(c)−1) = min priority strictly below c in its stack
        below = stacks[key][:-1]
        dd_below = min(below) if below else (n_total + 1)
        if dd_below <= max_n:
            continue
        # c ∈ Tw: add all valid relocations
        for dst in all_keys:
            if dst != key and len(stacks[dst]) < max_tiers:
                Ow.append((c, key, dst))

    # Cw = T ∪ Ow
    return T + Ow


# ─────────────────────────────────────────────────────────────────────────────
# Greedy warm-start (Gre-C, Section 4.2)
# ─────────────────────────────────────────────────────────────────────────────

def run_greedy_ubrp(
    stacks_init:       Dict[Any, List[int]],
    all_keys:          List[Any],
    n_total:           int,
    max_tiers:         int,
    key_to_1based_idx: Dict[Any, int],
) -> Tuple[List[Tuple[int, int, int, int]], int]:
    """
    Gre-C greedy for uBRP.

    Returns
    -------
    solution : list of (c, d_star, mc, t) 4-tuples for pheromone initialisation
    cost     : total number of relocations
    """
    stacks: Dict[Any, List[int]] = {k: list(v) for k, v in stacks_init.items()}
    M: Dict[int, int] = {}
    solution: List[Tuple[int, int, int, int]] = []

    # ── Build loc and smin ────────────────────────────────────────────── #
    loc: Dict[int, Optional[Any]] = {}
    smin: Dict[Any, int] = {}
    for key, prios in stacks.items():
        smin[key] = min(prios) if prios else (n_total + 1)
        for p in prios:
            loc[p] = key

    def _rebuild_smin(key: Any) -> None:
        smin[key] = min(stacks[key]) if stacks[key] else (n_total + 1)

    for target in range(1, n_total + 1):
        src_key = loc.get(target)
        if src_key is None:
            continue

        safety = 0
        while stacks[src_key] and stacks[src_key][-1] != target:
            safety += 1
            if safety > n_total * n_total:
                break   # pathological guard

            cands = build_candidates_grec(
                stacks, loc, smin, all_keys, src_key, target, n_total, max_tiers
            )
            if not cands:
                break

            # Pick candidate with minimum dife score (Eq. 24)
            best_c, best_src, best_dst = min(
                cands,
                key=lambda cd: _dife_score(cd[0], cd[1], cd[2], stacks, smin, n_total),
            )

            # Record 4-tuple
            sm_dst = smin[best_dst]
            d_val  = (sm_dst if sm_dst <= n_total
                      else n_total + key_to_1based_idx[best_dst])
            mc = M.get(best_c, 0)
            solution.append((best_c, d_val, mc, target))

            # Apply relocation
            stacks[best_src].pop()
            _rebuild_smin(best_src)
            stacks[best_dst].append(best_c)
            _rebuild_smin(best_dst)
            loc[best_c] = best_dst
            M[best_c]   = mc + 1

        # Retrieve target
        if stacks[src_key] and stacks[src_key][-1] == target:
            stacks[src_key].pop()
            _rebuild_smin(src_key)
            loc[target] = None

    return solution, len(solution)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────

def _dife_score(
    c:       int,
    src_key: Any,
    dst_key: Any,
    stacks:  Dict[Any, List[int]],
    smin:    Dict[Any, int],
    n_total: int,
) -> int:
    """
    Evaluate candidate (c → dst_key) using dife (Eq. 21) for greedy selection.
    Uses dd(S) = min priority (N+1 for empty) — not dd* — as the destination
    descriptor for the heuristic evaluation.
    """
    sm   = smin[dst_key]
    d_dd = sm if sm <= n_total else (n_total + 1)
    base = dif(c, d_dd, n_total)

    if smin[src_key] == c:          # c is the minimum → well-located in src
        below    = stacks[src_key][:-1]
        dd_below = min(below) if below else (n_total + 1)
        imp_c    = dd_below - c     # always ≥ 0 since c is WL
        return n_total + base - imp_c
    return base
