"""
SDH, VRH, and beam-search helpers for TingWuBS.

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
# Primitive helpers
# ─────────────────────────────────────────────────────────────────────────────

def fs_value(prios: List[int], n_total: int) -> int:
    """fs(s) = min priority in stack s (most urgent). Empty → N+1."""
    return min(prios) if prios else n_total + 1


def dis_value(b: int, prios: List[int], n_total: int) -> int:
    """DIs = fs(s) − b.  Positive means placing b in s creates no deadlock."""
    return fs_value(prios, n_total) - b


# ─────────────────────────────────────────────────────────────────────────────
# SDH: Smallest Difference Heuristic (≡ Caserta Min–Max, Section 4.1)
# ─────────────────────────────────────────────────────────────────────────────

def sdh_select(
    b:         int,
    src_key:   Any,
    stacks:    Dict[Any, List[int]],
    n_total:   int,
    max_tiers: int,
    all_keys:  List[Any],
) -> Optional[Any]:
    """
    Select destination for blocking container b using SDH.

    Rule:
      • ∃ s with DIs > 0  →  pick s with smallest DIs  (tightest no-deadlock)
      • All DIs ≤ 0        →  pick s with largest DIs   (delay re-relocation)
    """
    best_good_key: Optional[Any] = None
    best_good_di  = float("inf")
    best_bad_key: Optional[Any] = None
    best_bad_di   = float("-inf")

    for key in all_keys:
        if key == src_key:
            continue
        if len(stacks[key]) >= max_tiers:
            continue
        di = dis_value(b, stacks[key], n_total)
        if di > 0:
            if di < best_good_di:
                best_good_di  = di
                best_good_key = key
        else:
            if di > best_bad_di:
                best_bad_di  = di
                best_bad_key = key

    return best_good_key if best_good_key is not None else best_bad_key


def sdh_simulate(
    stacks_in: Dict[Any, List[int]],
    n_total:   int,
    max_tiers: int,
    all_keys:  List[Any],
) -> int:
    """
    Run the SDH greedy algorithm to completion.
    Returns total number of relocations (upper bound estimate).
    """
    stacks = {k: list(v) for k, v in stacks_in.items()}
    total  = 0

    for target in range(1, n_total + 1):
        src_key = next((k for k, p in stacks.items() if target in p), None)
        if src_key is None:
            continue

        while stacks[src_key] and stacks[src_key][-1] != target:
            blocker = stacks[src_key][-1]
            dst = sdh_select(blocker, src_key, stacks, n_total, max_tiers, all_keys)
            if dst is None:
                break
            stacks[src_key].pop()
            stacks[dst].append(blocker)
            total += 1

        if stacks[src_key] and stacks[src_key][-1] == target:
            stacks[src_key].pop()

    return total


# ─────────────────────────────────────────────────────────────────────────────
# VRH: Virtual Relocation Heuristic (Section 4.2)
# ─────────────────────────────────────────────────────────────────────────────

def _vrh_assign_blockers(
    blockers_bt: List[int],          # bottom-to-top priorities of blockers
    src_key:     Any,
    stacks:      Dict[Any, List[int]],
    n_total:     int,
    max_tiers:   int,
    all_keys:    List[Any],
) -> Dict[int, Any]:
    """
    Run Phase I + Phase II of VRH to assign each blocker index → destination.

    Parameters
    ----------
    blockers_bt : priorities of blocking containers above the target,
                  index 0 = directly above target (lowest tier),
                  index n−1 = topmost (highest tier, moved first physically).

    Returns
    -------
    assignments : {tier_index: dest_key}
    """
    n = len(blockers_bt)
    assignments: Dict[int, Any] = {}

    # -- shared tracking -------------------------------------------------------
    # fs_v[key]   : effective min priority of key in the virtual plan
    # Ts_v[key]   : min tier_idx of blockers already assigned to key (Phase I)
    # assigned_to[key]: list of (priority, tier_idx) already assigned to key
    fs_v        = {k: fs_value(stacks[k], n_total) for k in all_keys}
    Ts_v        = {k: float("inf") for k in all_keys}
    assigned_to: Dict[Any, List[Tuple[int, int]]] = {k: [] for k in all_keys}
    # virtual heights (original + assigned so far)
    vh = {k: len(stacks[k]) for k in all_keys}

    # ── Phase I ───────────────────────────────────────────────────────────────
    # Process blockers in descending priority order (largest number = least urgent first)
    phase1_order = sorted(range(n), key=lambda i: -blockers_bt[i])

    for i in phase1_order:
        b  = blockers_bt[i]   # priority
        tb = i                 # tier index (0 = closest to target)

        # Find valid stacks (Cond 1 AND Cond 2)
        valid: List[Tuple[int, Any]] = []
        for key in all_keys:
            if key == src_key:
                continue
            if vh[key] >= max_tiers:
                continue
            di = fs_v[key] - b          # Cond 1: di > 0
            ts = Ts_v[key]              # Cond 2: tb < Ts
            if di > 0 and tb < ts:
                valid.append((di, key))

        if valid:
            valid.sort(key=lambda x: x[0])   # smallest DIs first
            _, dst = valid[0]
            assignments[i] = dst
            # update virtual state
            fs_v[dst] = b               # b < old fs_v → new min = b
            Ts_v[dst] = min(Ts_v[dst], tb)
            assigned_to[dst].append((b, tb))
            vh[dst] += 1

    # ── Phase II ──────────────────────────────────────────────────────────────
    # Skipping containers: not placed in Phase I
    skipping = [i for i in range(n) if i not in assignments]
    # Process ascending priority (most urgent first = reverse of Phase I sort order)
    skipping.sort(key=lambda i: blockers_bt[i])

    for i in skipping:
        b  = blockers_bt[i]
        tb = i

        best_key: Optional[Any] = None
        best_vri  = float("-inf")

        for key in all_keys:
            if key == src_key:
                continue
            if vh[key] >= max_tiers:
                continue

            # Upper containers: assigned to key, tier_idx < tb, priority > b
            # (they are physically placed AFTER b → end up ABOVE b in dest)
            uppers = [(p, t) for p, t in assigned_to[key]
                      if t < tb and p > b]
            # Lower containers: assigned to key with tier_idx > tb + original containers
            lowers_assigned = [(p, t) for p, t in assigned_to[key] if t > tb]
            lowers = lowers_assigned + [(p, -1) for p in stacks[key]]

            U  = len(uppers)
            us = min(p for p, _ in uppers) if uppers else n_total + 1
            ls = min(p for p, _ in lowers) if lowers else n_total + 1

            if U <= 1:
                vri = min(ls - b, b - us)
            else:
                vri = -n_total + (b - us)

            if vri > best_vri:
                best_vri = vri
                best_key = key

        # Fallback to SDH if VRI couldn't find a valid key
        if best_key is None:
            best_key = sdh_select(b, src_key, stacks, n_total, max_tiers, all_keys)

        if best_key is not None:
            assignments[i] = best_key
            assigned_to[best_key].append((b, tb))
            fs_v[best_key] = min(fs_v[best_key], b)
            vh[best_key] += 1

    return assignments


def vrh_simulate(
    stacks_in: Dict[Any, List[int]],
    n_total:   int,
    max_tiers: int,
    all_keys:  List[Any],
) -> int:
    """
    Run the VRH greedy algorithm to completion.
    Returns total number of relocations (upper bound estimate).

    For each target container:
      1. Collect all blockers above it.
      2. Apply Phase I + II to assign each blocker a destination.
      3. Execute the planned moves top-to-bottom.
      4. Retrieve the target.
    """
    stacks = {k: list(v) for k, v in stacks_in.items()}
    total  = 0

    for target in range(1, n_total + 1):
        src_key = next((k for k, p in stacks.items() if target in p), None)
        if src_key is None:
            continue

        t_pos = stacks[src_key].index(target)
        blockers_bt = stacks[src_key][t_pos + 1:]   # bottom-to-top above target

        if not blockers_bt:
            stacks[src_key].pop()   # target already on top
            continue

        # Get VRH assignments (tier_index → dest_key)
        assignments = _vrh_assign_blockers(
            blockers_bt, src_key, stacks, n_total, max_tiers, all_keys
        )

        # Execute: top-to-bottom = highest tier_idx first
        n_blockers = len(blockers_bt)
        for i in range(n_blockers - 1, -1, -1):
            dst = assignments.get(i)
            if dst is None:
                # Emergency fallback: SDH
                dst = sdh_select(
                    blockers_bt[i], src_key, stacks, n_total, max_tiers, all_keys
                )
            if dst is not None:
                stacks[src_key].pop()
                stacks[dst].append(blockers_bt[i])
                total += 1

        # Retrieve target
        if stacks[src_key] and stacks[src_key][-1] == target:
            stacks[src_key].pop()

    return total


# ─────────────────────────────────────────────────────────────────────────────
# Beam Search helpers
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_accessible(
    stacks:  Dict[Any, List[int]],
    n_total: int,
) -> Dict[Any, List[int]]:
    """
    Retrieve every accessible target (in priority order) from the stacks.
    Returns a new stacks dict with those containers removed.
    """
    stacks = {k: list(v) for k, v in stacks.items()}
    for t in range(1, n_total + 1):
        src = next((k for k, p in stacks.items() if t in p), None)
        if src is None:
            continue   # already retrieved
        if stacks[src] and stacks[src][-1] == t:
            stacks[src].pop()
        else:
            break       # t is blocked → stop
    return stacks


def is_empty(stacks: Dict[Any, List[int]]) -> bool:
    """True iff all stacks are empty (bay is fully retrieved)."""
    return all(len(v) == 0 for v in stacks.values())


def get_top_blocker(
    stacks:  Dict[Any, List[int]],
    n_total: int,
) -> Tuple[Optional[int], Optional[Any]]:
    """
    Return (top_blocker_priority, src_key) for the current target container.
    Returns (None, None) if bay is empty or target already on top.
    """
    for t in range(1, n_total + 1):
        src = next((k for k, p in stacks.items() if t in p), None)
        if src is None:
            continue
        if stacks[src][-1] != t:
            return stacks[src][-1], src
        # target on top → will be auto-retrieved
    return None, None


def beam_search(
    stacks_init: Dict[Any, List[int]],
    n_total:     int,
    max_tiers:   int,
    all_keys:    List[Any],
    beam_width:  int,
    evaluator:   str = "VRH",
) -> int:
    """
    Ting & Wu (2017) Beam Search for CRP-R.

    Parameters
    ----------
    stacks_init : initial bay state
    beam_width  : b in the paper
    evaluator   : "VRH" (default) or "SDH"

    Returns
    -------
    Total number of relocations in the best solution found.
    """
    eval_fn = vrh_simulate if evaluator.upper() == "VRH" else sdh_simulate

    # Each beam state: (stacks_dict, parent_index)
    beam: List[Dict[Any, List[int]]] = [
        retrieve_accessible(stacks_init, n_total)
    ]

    level = 0   # = number of relocations performed

    while beam:
        # Check if any beam node has an empty bay → done
        if any(is_empty(s) for s in beam):
            return level

        # ── Expand: one relocation per beam node ──────────────────────────── #
        all_children: List[Tuple[Dict, float, int]] = []   # (stacks, eval, parent_idx)

        for parent_idx, state in enumerate(beam):
            blocker, src_key = get_top_blocker(state, n_total)
            if blocker is None:
                # Bay is empty or target accessible after auto-retrieve
                all_children.append((state, 0.0, parent_idx))
                continue

            for dst_key in all_keys:
                if dst_key == src_key:
                    continue
                if len(state[dst_key]) >= max_tiers:
                    continue

                # Create child state: one relocation
                child = {k: list(v) for k, v in state.items()}
                child[src_key].pop()
                child[dst_key].append(blocker)
                # Auto-retrieve accessible containers after this relocation
                child = retrieve_accessible(child, n_total)

                # Evaluate: estimate future relocations from child state
                est = float(eval_fn(child, n_total, max_tiers, all_keys))
                all_children.append((child, est, parent_idx))

        if not all_children:
            break

        level += 1

        # ── Select top-b nodes (dependent + independent tie-breaking) ─────── #
        all_children.sort(key=lambda x: x[1])

        if len(all_children) <= beam_width:
            beam = [c[0] for c in all_children]
        else:
            threshold = all_children[beam_width - 1][1]

            # Strictly better than threshold
            selected = [c for c in all_children if c[1] < threshold]

            # At threshold: independent strategy → diverse parents
            at_thresh = [c for c in all_children if c[1] == threshold]
            remaining = beam_width - len(selected)

            seen_parents = {c[2] for c in selected}
            from_new  = [c for c in at_thresh if c[2] not in seen_parents]
            from_seen = [c for c in at_thresh if c[2] in seen_parents]
            selected.extend((from_new + from_seen)[:remaining])

            beam = [c[0] for c in selected]

    return level
