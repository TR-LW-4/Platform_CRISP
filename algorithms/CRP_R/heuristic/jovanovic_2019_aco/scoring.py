"""
Greedy Min–Max and attractiveness helpers for JovanovicACO_rBRP.

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
# Stack value helpers
# ─────────────────────────────────────────────────────────────────────────────

def dd(stack_priorities: List[int], n_total: int) -> int:
    """
    dd(S) — minimum due-date (most urgent container) in stack S.
    Empty stack → N + 1  (sentinel: no container, never blocks anything).
    """
    if not stack_priorities:
        return n_total + 1
    return min(stack_priorities)


def dd_star(stack_priorities: List[int], n_total: int, stack_1based_idx: int) -> int:
    """
    dd*(S) — extended dd for pheromone-matrix indexing.

    Non-empty stack: returns dd(S) = min due-date ∈ {1..N}.
    Empty stack    : returns N + stack_1based_idx ∈ {N+1 .. N+W}.
      This differentiates empty stacks from each other so the pheromone
      matrix can learn which empty slot is most beneficial.
    """
    if not stack_priorities:
        return n_total + stack_1based_idx
    return min(stack_priorities)


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic scoring
# ─────────────────────────────────────────────────────────────────────────────

def dif(c: int, d: int, n_total: int) -> int:
    """
    Undesirability of placing container c above a stack whose
    minimum due-date is d.

    dif(c, d) ≤ N  →  c will be well-located (d > c, d retrieved after c)
    dif(c, d) > N  →  c will NOT be well-located (d < c, d retrieved before c)

    Range: 1 .. 2N.  Smaller = more desirable destination.

    Well-located case (d > c):
      dif = d − c  ← how many "slots" are wasted between c and d;
                     smaller = tighter fit = more future room preserved.

    Not-well-located case (d < c):
      dif = 2N + 1 − d  ← larger d → smaller dif → prefer stacks with
                          higher min-due-date (delays re-relocation longest).
    """
    if d > c:
        return d - c
    else:
        return 2 * n_total + 1 - d


def f_heuristic(c: int, d_val: int, n_total: int) -> float:
    """
    ACO heuristic attractiveness of placing c in a stack with
    dd* = d_val.  Inverse of (1 + dif) so that higher is more attractive.
    """
    return 1.0 / (1.0 + dif(c, d_val, n_total))


# ─────────────────────────────────────────────────────────────────────────────
# Lower bound
# ─────────────────────────────────────────────────────────────────────────────

def compute_lb(stacks: Dict[Any, List[int]]) -> int:
    """
    Lower bound on remaining relocations: number of non-well-located
    containers in the current bay state.

    Container c (at tier i, bottom-to-top index) is non-well-located if
    ∃ d at tier j < i (below c) with d < c.
    (d is retrieved before c but c is blocking d → c needs ≥ 1 relocation.)
    """
    lb = 0
    for prios in stacks.values():
        for i in range(1, len(prios)):
            c = prios[i]
            if min(prios[:i]) < c:
                lb += 1
    return lb


# ─────────────────────────────────────────────────────────────────────────────
# Greedy initialisation (MinMax heuristic, rBRP)
# ─────────────────────────────────────────────────────────────────────────────

def run_greedy_rbrp(
    stacks_init:        Dict[Any, List[int]],
    all_keys:           List[Any],
    n_total:            int,
    max_tiers:          int,
    key_to_1based_idx:  Dict[Any, int],
) -> Tuple[List[Tuple[int, int, int, int, Any]], int]:
    """
    Greedy rBRP using the MinMax heuristic (Eqs. 5/7).

    Returns
    -------
    solution : list of (c, d_star, mc, t, dst_key) tuples used to seed the
               pheromone matrix, where:
                 c      = due-date of the relocated container
                 d_star = dd*(destination stack) at the time of relocation
                 mc     = number of prior relocations of c (before this one)
                 t      = current target due-date
                 dst_key = physical destination stack (for final validation)
    cost     : total number of relocations
    """
    stacks: Dict[Any, List[int]] = {k: list(v) for k, v in stacks_init.items()}
    M: Dict[int, int] = {}                            # M[c] = relocation count
    solution: List[Tuple[int, int, int, int, Any]] = []

    for target in range(1, n_total + 1):
        src_key = next((k for k, p in stacks.items() if target in p), None)
        if src_key is None:
            continue

        # Relocate all blockers above target (top-first)
        while stacks[src_key] and stacks[src_key][-1] != target:
            c = stacks[src_key][-1]

            # MinMax: pick destination with minimum dif(c, dd(S))
            best_dst: Optional[Any] = None
            best_score = float("inf")
            for dst_key in all_keys:
                if dst_key == src_key:
                    continue
                if len(stacks[dst_key]) >= max_tiers:
                    continue
                score = dif(c, dd(stacks[dst_key], n_total), n_total)
                if score < best_score:
                    best_score = score
                    best_dst   = dst_key

            if best_dst is None:
                break  # no room — pathological case

            # Record 4-tuple for pheromone initialisation
            d_val = dd_star(stacks[best_dst], n_total, key_to_1based_idx[best_dst])
            mc    = M.get(c, 0)
            solution.append((c, d_val, mc, target, best_dst))

            stacks[src_key].pop()
            stacks[best_dst].append(c)
            M[c] = mc + 1

        # Retrieve target
        if stacks[src_key] and stacks[src_key][-1] == target:
            stacks[src_key].pop()

    return solution, len(solution)
