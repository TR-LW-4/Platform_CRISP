"""
MinMax heuristic and pheromone helpers for JovanovicACO_CRPTime.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from core.objectives import ObjectiveSpec


# ─────────────────────────────────────────────────────────────────────────────
# Stack value helpers  (copied from CRP_R/heuristic/jovanovic_2019_aco/scoring.py)
# ─────────────────────────────────────────────────────────────────────────────

def dd(stack_priorities: List[int], n_total: int) -> int:
    """
    dd(S) — minimum due-date (most urgent container) in stack S.  Eq. 2.
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
      This differentiates empty stacks so the pheromone matrix can learn
      which empty slot is most beneficial.
    """
    if not stack_priorities:
        return n_total + stack_1based_idx
    return min(stack_priorities)


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic scoring  (copied — unchanged for crane-time; per paper Section 6)
# ─────────────────────────────────────────────────────────────────────────────

def dif(c: int, d: int, n_total: int) -> int:
    """
    Eq. 3 — undesirability of placing container c above a stack whose
    minimum due-date is d.

    The MinMax heuristic function is kept identical for the crane-time
    variant (Section 6 of the paper uses the same heuristic; only val()
    and the lower bound change).

    Range: 1 .. 2N.  Smaller = more desirable destination.
    """
    if d > c:
        return d - c
    else:
        return 2 * n_total + 1 - d


# ─────────────────────────────────────────────────────────────────────────────
# Crane-time lower bounds LB_f and LB_v  (Sec. 6, eqs. 44–45, p. 85)
# ─────────────────────────────────────────────────────────────────────────────

def has_time_lb(spec: ObjectiveSpec) -> bool:
    """The paper defines a crane-time bound only for O_f (f2) and O_v (f2vert)."""
    return spec.mode == "crane_time" and spec.time_model in ("f2", "f2_vertical")


def lb_container(s: int, t: int, nw: bool, spec: ObjectiveSpec) -> float:
    """
    Contribution of one container at stack s, tier t to LB_f (eq. 44) or
    LB_v (eq. 45): its retrieval from the current position, plus the
    minimal relocation cost if it is non-well-located (nw).

    Taken literally from the paper; LB_f is not always a valid lower bound
    (see docs/validation/jovanovic_2019.md §3.6).
    """
    ts = spec.stack_s_per_stack
    if spec.time_model == "f2":
        tpp = spec.pickup_place_s
        return tpp + 2.0 * s * ts + ((tpp + ts) if nw else 0.0)
    tr = spec.empty_vertical_s_per_tier + spec.loaded_vertical_s_per_tier
    h_max = spec.max_tiers + 1
    return (
        (2.0 * h_max - t - spec.outside_height) * tr
        + 2.0 * s * ts
        + ((2.0 * tr + ts) if nw else 0.0)
    )


def lb_time(
    stacks: Dict[Any, List[int]],
    key_to_1based_idx: Dict[Any, int],
    spec: ObjectiveSpec,
) -> float:
    """LB_f or LB_v of a bay (tiers 1-based from the ground); 0 without a paper bound."""
    if not has_time_lb(spec):
        return 0.0
    total = 0.0
    for key, prios in stacks.items():
        below_min = None
        for i, c in enumerate(prios):
            nw = below_min is not None and below_min < c
            total += lb_container(key_to_1based_idx[key], i + 1, nw, spec)
            below_min = c if below_min is None else min(below_min, c)
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Greedy start S_g  (Alg. 1, p. 80)
# ─────────────────────────────────────────────────────────────────────────────

def run_greedy_rbrp_time(
    stacks_init:       Dict[Any, List[int]],
    all_keys:          List[Any],
    n_total:           int,
    max_tiers:         int,
    key_to_1based_idx: Dict[Any, int],
    max_moves:         int,
) -> List[Tuple[int, int, int, int]]:
    """
    MinMax greedy for the rBRP (Alg. 1). Returns S_g as (c, dd*(S), n, t)
    4-tuples; its cost is evaluated by the caller with the shared evaluator.
    """
    stacks: Dict[Any, List[int]] = {k: list(v) for k, v in stacks_init.items()}
    M: Dict[int, int] = {}
    solution: List[Tuple[int, int, int, int]] = []

    for target in range(1, n_total + 1):
        src_key: Optional[Any] = next(
            (k for k, p in stacks.items() if target in p), None
        )
        if src_key is None:
            continue

        # Relocate all blockers above target (top-first), MinMax destination
        while stacks[src_key] and stacks[src_key][-1] != target:
            c = stacks[src_key][-1]

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

            # R_c is empty (eq. 1): Alg. 1 cannot continue, so S_g and
            # tau_0 (eq. 37) are undefined.
            if best_dst is None:
                raise ValueError(
                    f"Jovanović greedy start (Alg. 1): no free stack for container "
                    f"{c} above target {target}; every other stack is full"
                )

            d_val = dd_star(stacks[best_dst], n_total, key_to_1based_idx[best_dst])
            n     = M.get(c, 0)
            # n = 0, ..., MaxMoves (p. 83); M keeps the true count, as for the ants
            solution.append((c, d_val, min(n, max_moves), target))

            stacks[src_key].pop()
            stacks[best_dst].append(c)
            M[c] = n + 1

        # Retrieve target
        if stacks[src_key] and stacks[src_key][-1] == target:
            stacks[src_key].pop()

    return solution
