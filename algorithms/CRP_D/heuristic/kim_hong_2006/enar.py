"""
ENAR (Expected Number of Additional Relocations) for group-priority BRP.

Implements:
  - E(k, n)  via Equation (2) in Kim & Hong (2006), §3.2
  - enar_stack()   total ENAR for one stack given current bay state
  - enar_bay()     sum of enar_stack() over all stacks

Reference
---------
K.H. Kim and G.-P. Hong,
"A heuristic rule for relocating blocks",
Computers & Operations Research 33 (2006) 940–954.

Notation (from paper)
---------------------
  N   : total blocks currently in bay
  T   : expected maximum stack height (= average current height, ≥ 1)
  n   : "highest priority number" = highest group code currently in stack,
        i.e. the group-code whose container would be retrieved last
        (larger group code = retrieved later)
  k   : number of empty slots in the stack (T − actual height; ≥ 0)
  ni  : number of blocks with group code i remaining in the bay,
        EXCLUDING blocks in the stack under consideration
  E(k, n) : expected additional relocations from k empty slots in a stack
            whose highest group code is n

Equation (2):
  E(0, n) = 0
  E(k, n) = [Σ_{i=1}^{n} ni·E(k-1,i) + (N-T+k - Σ_{i=1}^{n} ni)·(1+E(k-1,n))]
            / (N - T + k)

T for a given stack is computed as:
  max(1, average height of all stacks that are non-empty,
         including a 1-slot allowance for full stacks)

For stacks that are ALREADY at or above T:
  we still allow exactly 1 virtual empty slot (T_eff = actual_height + 1),
  unless the stack is at the hard capacity limit (is_full), in which case
  it is excluded (ENAR = 0 and not a candidate destination).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Tuple


# ------------------------------------------------------------------ #
# Public API                                                          #
# ------------------------------------------------------------------ #

def compute_enar_bay(
    stacks_groups: List[List[int]],  # each inner list = group codes bottom→top
    max_tiers: int,
) -> List[float]:
    """
    Compute ENAR for every stack in the bay.

    Parameters
    ----------
    stacks_groups : list of lists, each giving group codes bottom→top
    max_tiers     : hard height capacity (stacks at this height are excluded)

    Returns
    -------
    List[float] of same length as stacks_groups; ENAR = 0 for excluded stacks.
    """
    heights = [len(s) for s in stacks_groups]
    N       = sum(heights)
    S       = len(stacks_groups)

    if N == 0:
        return [0.0] * S

    # T = average height (≥ 1); Paper: "average height of stacks in bay"
    non_empty_heights = [h for h in heights if h > 0]
    T = max(1.0, sum(non_empty_heights) / max(len(non_empty_heights), 1))

    results: List[float] = []
    for s_idx, stk in enumerate(stacks_groups):
        h = heights[s_idx]
        if h >= max_tiers:
            # Full stack: excluded from relocation targets; ENAR = 0
            results.append(0.0)
            continue

        # n = highest group code in this stack (later retrieved = larger number)
        if stk:
            n = max(stk)
        else:
            results.append(0.0)
            continue

        # T_eff for this stack:
        # if h >= T (taller than average): allow 1 virtual slot
        T_eff = T if h < T else float(h + 1)
        k = max(0, int(round(T_eff)) - h)
        if k == 0:
            # No empty slots to fill → no additional relocations expected
            results.append(0.0)
            continue

        # ni: group code counts in the bay EXCLUDING this stack
        ni_map: Dict[int, int] = {}
        for s2, stk2 in enumerate(stacks_groups):
            if s2 == s_idx:
                continue
            for g in stk2:
                ni_map[g] = ni_map.get(g, 0) + 1

        val = _enar_stack(k, n, N, int(round(T_eff)), ni_map)
        results.append(val)

    return results


def enar_after_place(
    stacks_groups: List[List[int]],
    max_tiers: int,
    dst_idx: int,
    placed_group: int,
) -> float:
    """
    ENAR of stack `dst_idx` after placing a container of `placed_group` on top.
    Used to evaluate E(s'_j) in R[a] = E(S') - E(S) + r[a].
    """
    new_stacks = [list(s) for s in stacks_groups]
    new_stacks[dst_idx] = new_stacks[dst_idx] + [placed_group]
    vals = compute_enar_bay(new_stacks, max_tiers)
    return vals[dst_idx]


# ------------------------------------------------------------------ #
# Internal recursive computation (Equation 2)                        #
# ------------------------------------------------------------------ #

def _enar_stack(
    k: int,
    n: int,
    N: int,
    T: int,
    ni_map: Dict[int, int],
) -> float:
    """
    Compute E(k, n) using Equation (2).

    k       : number of empty slots
    n       : highest group code in the stack (blocks of group n retrieved last)
    N       : total blocks in bay
    T       : expected max stack height (integer approximation)
    ni_map  : {group_code: count_in_bay_excluding_this_stack}
    """
    # Use a memo dict keyed by (k, n) for this call tree
    memo: Dict[Tuple[int, int], float] = {}
    return _E(k, n, N, T, ni_map, memo)


def _E(
    k: int,
    n: int,
    N: int,
    T: int,
    ni_map: Dict[int, int],
    memo: Dict[Tuple[int, int], float],
) -> float:
    if k == 0:
        return 0.0
    key = (k, n)
    if key in memo:
        return memo[key]

    denom = N - T + k
    if denom <= 0:
        memo[key] = 0.0
        return 0.0

    # Sum_{i=1}^{n} ni * E(k-1, i)
    sum_ni_E: float = 0.0
    sum_ni: int = 0
    for i in range(1, n + 1):
        ni = ni_map.get(i, 0)
        if ni > 0:
            sum_ni += ni
            sum_ni_E += ni * _E(k - 1, i, N, T, ni_map, memo)

    # Remaining (groups with code > n or not yet in bay): contribute (1 + E(k-1, n))
    remaining = denom - sum_ni
    if remaining < 0:
        remaining = 0

    val = (sum_ni_E + remaining * (1.0 + _E(k - 1, n, N, T, ni_map, memo))) / denom
    memo[key] = val
    return val
