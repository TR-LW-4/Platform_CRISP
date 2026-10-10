"""
Greedy start S_g of JovanovicACO_CRPTime (Alg. 1, p. 80): every blocker goes to
the stack the MinMax heuristic prefers. Its solution starts the colony and sets
the initial pheromone.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from algorithms.CRP_Time.utils.bay import target_stack
from algorithms.CRP_Time.utils.located import min_due

from .minmax import dd_star, dif


def greedy(
    stacks_init: List[List[int]],
    n_total:     int,
    max_tiers:   int,
    max_moves:   int,
) -> List[Tuple[int, int, int, int]]:
    """
    MinMax greedy for the rBRP (Alg. 1). Returns S_g as (c, dd*(S), n, t)
    4-tuples; its cost is evaluated by the caller with the shared evaluator.
    """
    stacks = [list(st) for st in stacks_init]
    M: Dict[int, int] = {}
    solution: List[Tuple[int, int, int, int]] = []

    for target in range(1, n_total + 1):
        src = target_stack(stacks, target)
        if src < 0:
            continue

        # Relocate all blockers above target (top-first), MinMax destination
        while stacks[src] and stacks[src][-1] != target:
            c = stacks[src][-1]

            best_dst = -1
            best_score = float("inf")
            for dst in range(len(stacks)):
                if dst == src:
                    continue
                if len(stacks[dst]) >= max_tiers:
                    continue
                score = dif(c, min_due(stacks[dst], n_total + 1), n_total)
                if score < best_score:
                    best_score = score
                    best_dst   = dst

            # R_c is empty (eq. 1): Alg. 1 cannot continue, so S_g and
            # tau_0 (eq. 37) are undefined.
            if best_dst < 0:
                raise ValueError(
                    f"Jovanović greedy start (Alg. 1): no free stack for container "
                    f"{c} above target {target}; every other stack is full"
                )

            d_val = dd_star(stacks[best_dst], n_total, best_dst + 1)
            n     = M.get(c, 0)
            # n = 0, ..., MaxMoves (p. 83); M keeps the true count, as for the ants
            solution.append((c, d_val, min(n, max_moves), target))

            stacks[src].pop()
            stacks[best_dst].append(c)
            M[c] = n + 1

        # Retrieve target
        if stacks[src] and stacks[src][-1] == target:
            stacks[src].pop()

    return solution
