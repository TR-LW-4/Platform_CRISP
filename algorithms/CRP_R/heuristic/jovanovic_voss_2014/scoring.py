"""
Jovanović & Voß (2014) Min–Max, Chain, and Chain-F heuristic scoring.

Notation follows the paper exactly:
  p(i)      = min priority value in stack i (= most urgent block, lowest number)
              Empty stack → N + 1
  d(i, r)   = p(i) − r
              > 0  ⟹  placing block r on stack i creates NO new deadlock
              ≤ 0  ⟹  a deadlock would be created

Min–Max (Eq. 4)
  • If ∃ i : d(i,r) > 0  →  s* = argmin_i d(i,r)  (no-deadlock stack, tightest fit)
  • Otherwise             →  s* = argmax_i d(i,r)  (least-bad deadlock)

Chain-F correction (Eq. 5, bad-case only)
  When placing block r on a stack i that would become FULL (height = max_tiers),
  substitute p(i) ← −N − p(i) so that such stacks are never chosen.

Extended Min–Max f*(r, se, Bay) (Eq. 6)
  Same as above but with an additional excluded stack se.

Chain heuristic (Eqs. 7–11)
  After computing sD = fMinMax(rn), if p(rn) < p(rn+1) evaluate the
  "reverse-order" scenario and optionally override sD with sR.

Reference
---------
R. Jovanović, S. Voß,
"A chain heuristic for the Blocks Relocation Problem",
Computers & Industrial Engineering 75 (2014) 79–86.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Primitive helpers
# ─────────────────────────────────────────────────────────────────────────────

def _p_stack(stk, n_total: int) -> int:
    """
    p(i) = min priority value in stack i.
    Empty stack returns N+1 (sentinel: no conflict with any block).
    Containers are stored bottom-to-top in stk.containers.
    """
    if stk is None or stk.is_empty:
        return n_total + 1
    return min(int(c.priority) for c in stk.containers)


# ─────────────────────────────────────────────────────────────────────────────
# Core Min–Max selection (Eqs. 4 & 6)
# ─────────────────────────────────────────────────────────────────────────────

def _min_max_core(
    env,
    r_priority:  int,
    src_key:     Tuple[int, int],
    extra_excl:  Optional[Tuple[int, int]] = None,
    use_chain_f: bool = False,
    max_tiers:   Optional[int] = None,
    p_override:  Optional[Dict[Tuple[int, int], int]] = None,
) -> Tuple[int, int]:
    """
    Select destination stack for a block of priority *r_priority*.

    Parameters
    ----------
    env         : CRP-R environment instance
    r_priority  : priority of the block being relocated
    src_key     : (bay, row) – source stack (always excluded from candidates)
    extra_excl  : optional additional stack to exclude  (Eq. 6 f* variant)
    use_chain_f : apply Chain-F full-stack correction in the bad (deadlock) case
    max_tiers   : must be provided when use_chain_f=True
    p_override  : stack_key → override for p(i); used to simulate the hypothetical
                  state after a prior move (for computing D2)

    Returns
    -------
    (best_action_index, d_value)
    """
    n_total  = env.config.num_containers
    n_stacks = env.config.num_bays * env.config.num_rows
    mask     = env._get_info().get("action_mask")

    def get_p(key: Tuple[int, int]) -> int:
        if p_override and key in p_override:
            return p_override[key]
        return _p_stack(env.yard.stacks.get(key), n_total)

    # Build candidate list: (action_index, stack_key, p_value)
    cands: List[Tuple[int, Tuple[int, int], int]] = []
    for a in range(n_stacks):
        if mask is not None and not mask[a]:
            continue
        key = env._action_to_stack(a)
        if key == src_key:
            continue
        if extra_excl is not None and key == extra_excl:
            continue
        cands.append((a, key, get_p(key)))

    if not cands:
        return 0, 0

    # Partition into good (d > 0, no deadlock) and bad (d ≤ 0, deadlock)
    good = [(a, key, p) for a, key, p in cands if p > r_priority]
    bad  = [(a, key, p) for a, key, p in cands if p <= r_priority]

    if good:
        # argmin d(i, r) = argmin p(i);  break ties by smaller action index
        best_a, _, best_p = min(good, key=lambda t: (t[2], t[0]))
    else:
        # argmax d(i, r) = argmax p(i);  Chain-F penalises would-be-full stacks
        def _bad_sort_key(t: Tuple[int, Tuple[int, int], int]):
            a, key, p = t
            if use_chain_f and max_tiers is not None:
                stk = env.yard.stacks.get(key)
                if stk is not None and len(stk.containers) == max_tiers - 1:
                    # Eq. 5: p'(i) = −N − p(i)  →  sort key becomes very small
                    return (-n_total - p, -a)
            return (p, -a)

        best_a, _, best_p = max(bad, key=_bad_sort_key)

    return best_a, best_p - r_priority   # d_value = p(i) − r


# ─────────────────────────────────────────────────────────────────────────────
# Identify rn+1  (the next block to be relocated)
# ─────────────────────────────────────────────────────────────────────────────

def _find_rn1(env) -> Tuple:
    """
    Return (rn1_container, rn1_src_key) for the block that will need to be
    relocated at the NEXT relocation step, or (None, None) if not found.

    rn+1 is:
    • the block immediately below the current top blocker (rn) in the same
      stack — when multiple blockers sit above the target, or
    • the top blocker of the NEXT target's stack — when rn sits directly on
      the current target.
    """
    target = env._get_target_container()
    if target is None:
        return None, None

    src = env.yard._find_stack(target)
    if src is None:
        return None, None

    src_key = (src.bay, src.row)
    containers = src.containers   # bottom-to-top list (Stack stores them so)

    # Locate target's index
    t_idx = None
    for i, c in enumerate(containers):
        if int(c.priority) == int(target.priority):
            t_idx = i
            break
    if t_idx is None:
        return None, None

    above = containers[t_idx + 1:]   # blocks from target upward; last = top = rn

    if len(above) >= 2:
        # rn+1 is the block just below rn
        return above[-2], src_key

    # rn is the only blocker → rn+1 is the top blocker of the next target
    next_priority = int(target.priority) + 1
    next_target = None
    for c in env.containers:
        if int(c.priority) == next_priority:
            next_target = c
            break
    if next_target is None:
        return None, None

    next_src = env.yard._find_stack(next_target)
    if next_src is None:
        return None, None

    # If next target is already accessible (no blockers), no rn+1 here
    if int(next_src.top.priority) == int(next_target.priority):
        return None, None

    return next_src.top, (next_src.bay, next_src.row)


# ─────────────────────────────────────────────────────────────────────────────
# Hypothetical p-value overrides for D2 computation
# ─────────────────────────────────────────────────────────────────────────────

def _p_override_after_move(
    env,
    rn_priority: int,
    src_key:     Tuple[int, int],
    sD_key:      Tuple[int, int],
) -> Dict[Tuple[int, int], int]:
    """
    Compute p(i) overrides for the hypothetical state where rn (priority
    rn_priority) has been moved from src_key to sD_key.

    Only src_key and sD_key change; all other stacks are unaffected.
    """
    n_total = env.config.num_containers
    overrides: Dict[Tuple[int, int], int] = {}

    # sD gains rn on top
    sD_stk  = env.yard.stacks.get(sD_key)
    old_sD  = _p_stack(sD_stk, n_total)
    overrides[sD_key] = min(old_sD, rn_priority)

    # src loses rn (was on top) — remove exactly one occurrence of rn_priority
    src_stk = env.yard.stacks.get(src_key)
    if src_stk and src_stk.containers:
        remaining = list(src_stk.containers)
        # rn was the top element; remove the last matching priority
        for i in range(len(remaining) - 1, -1, -1):
            if int(remaining[i].priority) == rn_priority:
                remaining.pop(i)
                break
        overrides[src_key] = (
            min(int(c.priority) for c in remaining) if remaining else n_total + 1
        )
    else:
        overrides[src_key] = n_total + 1

    return overrides


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def chain_select_action(env, use_chain_f: bool = True) -> int:
    """
    Jovanović & Voß (2014) Chain heuristic destination selection.

    Steps
    -----
    1. Compute sD = fMinMax(rn, Bayn)  and  D1 = d(sD, rn)   (Eqs. 4, 7)
    2. Identify rn+1; if p(rn) ≥ p(rn+1) → chain logic skipped, return sD
    3. R2 = d(sR*, rn+1)  where  sR* = fMinMax(rn+1, Bayn)   (Eq. 9)
    4. R1 = d(sR,  rn )   where  sR  = f*MinMax(rn, sR*, Bayn) (Eq. 10)
    5. D2 = d(sD*, rn+1)  where  sD* = fMinMax(rn+1, Bayn+1)   (Eq. 8)
       – Bayn+1 is simulated by adjusting p() values for src and sD
    6. Chain decision (Eq. 11):
       if  D1 > 0  and  R2 > 0  and  R2 < D1  and  R1·D2 > 0  → return sR
       else return sD

    Returns flat action index.
    """
    target = env._get_target_container()
    if target is None:
        return 0

    src = env.yard._find_stack(target)
    if src is None:
        return 0
    if int(src.top.priority) == int(target.priority):
        return 0   # target is already accessible

    rn          = src.top
    rn_priority = int(rn.priority)
    src_key     = (src.bay, src.row)
    max_tiers   = int(env.config.max_tiers)

    # ── Step 1 : base Min–Max choice for rn ──────────────────────────────────
    sD_action, D1 = _min_max_core(
        env, rn_priority, src_key,
        use_chain_f=use_chain_f, max_tiers=max_tiers,
    )

    # ── Step 2 : find rn+1 and check chain precondition ─────────────────────
    rn1, rn1_src_key = _find_rn1(env)

    # Paper: chain only applies when p(rn) < p(rn+1)  i.e. rn more urgent
    if rn1 is None or rn_priority >= int(rn1.priority):
        return sD_action

    rn1_priority = int(rn1.priority)

    # ── Step 3 : R2 — fMinMax(rn+1, Bayn) ───────────────────────────────────
    sR_star_action, R2 = _min_max_core(
        env, rn1_priority, rn1_src_key,
        use_chain_f=use_chain_f, max_tiers=max_tiers,
    )
    sR_star_key = env._action_to_stack(sR_star_action)

    # ── Step 4 : R1 — f*MinMax(rn, sR*, Bayn) ───────────────────────────────
    sR_action, R1 = _min_max_core(
        env, rn_priority, src_key,
        extra_excl=sR_star_key,
        use_chain_f=use_chain_f, max_tiers=max_tiers,
    )

    # ── Step 5 : D2 — fMinMax(rn+1, Bayn+1) ─────────────────────────────────
    sD_key  = env._action_to_stack(sD_action)
    p_ovrd  = _p_override_after_move(env, rn_priority, src_key, sD_key)
    _, D2   = _min_max_core(
        env, rn1_priority, rn1_src_key,
        use_chain_f=use_chain_f, max_tiers=max_tiers,
        p_override=p_ovrd,
    )

    # ── Step 6 : Chain decision (Eq. 11) ────────────────────────────────────
    # Use sR (reverse order) iff ALL three criteria satisfied:
    #   • D1 > 0 and R2 > 0  (neither move creates a new deadlock)
    #   • R2 < D1             (rn+1 prefers current state over post-rn state)
    #   • R1 * D2 > 0         (both "second moves" have the same deadlock sign)
    if D1 > 0 and R2 > 0 and R2 < D1 and R1 * D2 > 0:
        return sR_action

    return sD_action
