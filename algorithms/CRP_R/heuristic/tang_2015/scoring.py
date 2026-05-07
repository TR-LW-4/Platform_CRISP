"""
Tang, Jiang, Liu & Dong (2015) — H1 / H2 heuristic scoring.

Stack data is represented as Dict[key, List[int]] where
  key       = any hashable (e.g. (bay, row) tuple from the platform)
  List[int] = container priorities bottom-to-top; lower value = retrieved first

Paper notation mapping
----------------------
  container number k  →  priority value  (lower = earlier retrieval)
  nc(c)               →  nc_value(prios, n_total)  = min priority in stack
  RI(k, c)            →  ri_score(prios, k)  = |{p ∈ c : p < k}|
  BI(k, c)            →  bi_score(prios, k)  = containers above min after placing k

H1 rule (Eq. in Section 4.1.2)
  1. If ∃ c : nc(c) > k  →  put k in the column with the smallest nc > k
  2. Otherwise           →  put k in the column with the minimum RI
     (tie-break: largest nc)

H2 rule
  Same as H1 but step 2 uses minimum BI instead of minimum RI.

Extended variant (*-E)
  For each feasible destination, simulate the full remaining retrieval
  sequence using the original heuristic and pick the destination that
  minimises total reshuffles.  Ties resolved by the base H1/H2 rule.

Reference
---------
L. Tang, W. Jiang, J. Liu, Y. Dong,
"Research into container reshuffling and stacking problems in container
 terminal yards",
IIE Transactions, 47(7), 751–766, 2015.
https://doi.org/10.1080/0740817X.2014.971201
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Primitive scoring functions
# ─────────────────────────────────────────────────────────────────────────────

def nc_value(priorities: List[int], n_total: int) -> int:
    """
    nc(c) = smallest priority (most urgent) container in stack.
    Empty stack returns n_total + 1 (treated as 'never blocks anything').
    """
    return min(priorities) if priorities else n_total + 1


def ri_score(priorities: List[int], k: int) -> int:
    """
    RI(k, c) = number of containers in stack with priority < k.
    These containers are retrieved before k and would require k to be
    reshuffled when they are accessed.
    """
    return sum(1 for p in priorities if p < k)


def bi_score(priorities: List[int], k: int) -> int:
    """
    BI(k, c) = after placing k on top of stack c, the number of containers
    above the most-urgent container (min priority) in the resulting stack.

    If k becomes the most urgent (k < all existing), returns 0 (k is on top,
    nothing blocks it).
    """
    if not priorities:
        return 0
    new = priorities + [k]
    min_p = min(new)
    min_pos = new.index(min_p)   # first occurrence (bottom-to-top)
    return len(new) - 1 - min_pos


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stacks_from_env(env) -> Tuple[Dict[Any, List[int]], Dict[Any, int]]:
    """
    Convert env yard to a plain dict representation.

    Returns
    -------
    stacks       : {stack_key: [priority_bottom, ..., priority_top]}
    key_to_action: {stack_key: flat_action_index}
    """
    n_stacks = env.config.num_bays * env.config.num_rows
    stacks: Dict[Any, List[int]] = {}
    key_to_action: Dict[Any, int] = {}
    for a in range(n_stacks):
        key = env._action_to_stack(a)
        stk = env.yard.stacks.get(key)
        stacks[key] = [int(c.priority) for c in stk.containers] if stk else []
        key_to_action[key] = a
    return stacks, key_to_action


def _h1h2_select_key(
    k:         int,
    src_key:   Any,
    stacks:    Dict[Any, List[int]],
    n_total:   int,
    max_tiers: int,
    rule:      str,
) -> Optional[Any]:
    """
    Core H1 / H2 destination selection.

    Parameters
    ----------
    k         : priority of the blocking container being relocated
    src_key   : key of the source stack (excluded from candidates)
    stacks    : current stack state
    rule      : "H1" (RI-based) or "H2" (BI-based)

    Returns
    -------
    Destination stack key, or None if no valid destination exists.
    """
    cands = [
        (key, prios)
        for key, prios in stacks.items()
        if key != src_key and len(prios) < max_tiers
    ]
    if not cands:
        return None

    # Good stacks: nc > k → placing k here will not create a future deadlock
    good = [(key, prios) for key, prios in cands if nc_value(prios, n_total) > k]

    if good:
        # Among good stacks, pick nc closest to k (smallest nc > k).
        # This leaves stacks with larger nc available for later blockers.
        return min(good, key=lambda t: nc_value(t[1], n_total))[0]

    # No good stack — unavoidable deadlock; choose by secondary criterion.
    # Tie-break: largest nc (delays future re-relocation as long as possible).
    if rule == "H1":
        return min(
            cands,
            key=lambda t: (ri_score(t[1], k), -nc_value(t[1], n_total)),
        )[0]
    else:  # H2
        return min(
            cands,
            key=lambda t: (bi_score(t[1], k), -nc_value(t[1], n_total)),
        )[0]


# ─────────────────────────────────────────────────────────────────────────────
# Forward simulation (used by the Extended variant)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_from(
    stacks:        Dict[Any, List[int]],
    from_priority: int,
    n_total:       int,
    max_tiers:     int,
    rule:          str,
) -> int:
    """
    Simulate full retrieval from *from_priority* to *n_total* using H1/H2.
    The input dict is deep-copied internally.

    Returns total number of reshuffles.
    """
    s = {key: list(prios) for key, prios in stacks.items()}
    reshuffles = 0

    for target in range(from_priority, n_total + 1):
        # Locate the stack holding this target
        src_key = next(
            (key for key, prios in s.items() if target in prios),
            None,
        )
        if src_key is None:
            continue

        # Move all blockers above target (top-first) using H1/H2
        t_pos = s[src_key].index(target)
        while len(s[src_key]) > t_pos + 1:
            blocker = s[src_key][-1]   # topmost container
            dst_key = _h1h2_select_key(blocker, src_key, s, n_total, max_tiers, rule)
            if dst_key is None:
                break   # pathological — no space anywhere
            s[src_key].pop()
            s[dst_key].append(blocker)
            reshuffles += 1

        # Retrieve the target
        s[src_key].remove(target)

    return reshuffles


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def tang_select_action(env, rule: str = "H2", use_extended: bool = True) -> int:
    """
    Select the destination action for the current relocation step.

    Parameters
    ----------
    env          : CRP-R environment instance
    rule         : "H1" or "H2"
    use_extended : if True, run the Extended (*-E) look-ahead

    Returns
    -------
    Flat action index compatible with env.step().
    """
    target = env._get_target_container()
    if target is None:
        return 0

    src = env.yard._find_stack(target)
    if src is None:
        return 0
    if int(src.top.priority) == int(target.priority):
        return 0   # target already on top — will be auto-retrieved

    n_total    = int(env.config.num_containers)
    max_tiers  = int(env.config.max_tiers)
    t_priority = int(target.priority)
    k          = int(src.top.priority)   # the blocker to move
    src_key    = (src.bay, src.row)

    stacks, key_to_action = _stacks_from_env(env)

    # Build valid candidate destinations (respects action mask)
    info = env._get_info()
    mask = info.get("action_mask")
    n_stacks = env.config.num_bays * env.config.num_rows

    valid_keys: List[Any] = []
    for a in range(n_stacks):
        if mask is not None and not mask[a]:
            continue
        key = env._action_to_stack(a)
        if key != src_key:
            valid_keys.append(key)

    if not valid_keys:
        return 0

    # Baseline H1/H2 choice (always computed; used as tie-breaker in extended)
    h1h2_key = _h1h2_select_key(k, src_key, stacks, n_total, max_tiers, rule)

    if not use_extended:
        return key_to_action.get(h1h2_key, 0) if h1h2_key else 0

    # ── Extended variant: try every feasible destination ─────────────────────
    best_key   = h1h2_key
    best_cost  = float("inf")

    for dst_key in valid_keys:
        # Simulate: place k at dst_key, then run H1/H2 to completion
        trial = {key: list(prios) for key, prios in stacks.items()}
        trial[src_key].pop()        # remove k (it was the topmost)
        trial[dst_key].append(k)    # place at candidate destination

        cost = simulate_from(trial, t_priority, n_total, max_tiers, rule)

        if cost < best_cost:
            best_cost = cost
            best_key  = dst_key

    return key_to_action.get(best_key, 0) if best_key else 0
