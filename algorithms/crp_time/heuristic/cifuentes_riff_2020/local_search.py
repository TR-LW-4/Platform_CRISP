"""
G-CREM (Cifuentes & Riff, 2020) — Post-processing hill-climbing phase.

Reference
---------
C.D. Cifuentes and M.C. Riff,
"G-CREM: A GRASP approach to solve the container relocation problem
for multibays", Applied Soft Computing 97 (2020) 106721.

Algorithm
---------
Given a candidate plan produced by the constructive phase, the
hill-climbing procedure:

1. Identifies the container relocated the most times in the plan
   (ties broken by lower ``container_id``).  Already-tried containers
   within the current pass are skipped.
2. Locates that container's LAST relocation (its ``Movement`` whose
   ``to_pos`` is a stack, not ``None``, with the largest index).
3. Considers every other feasible destination for that last relocation.
   For each candidate destination the plan is "cut" at the edit point
   and rebuilt from there on with the RIL repair heuristic; if the
   resulting plan has a strictly smaller evaluation value it is
   accepted and the tried-set is cleared (improved plans may expose new
   improvable containers).
4. The procedure stops after ``max_iter`` iterations or when every
   relocated container has been tried without improvement.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Set, Tuple

from core.objectives import KinematicsModel, compute_crane_time
from core.plan       import Movement, RelocationPlan
from .ril            import ril_select_dst


# ================================================================ #
#  Evaluation function                                               #
# ================================================================ #

def eval_plan(
    plan:  RelocationPlan,
    kin:   KinematicsModel,
    alpha: float,
    beta:  float,
) -> float:
    """
    G-CREM weighted objective:  α · (# moves)  +  β · crane_time.

    ``# moves`` counts every Movement (relocations + retrievals) — since
    the number of retrievals is constant for a given instance, the
    ranking is equivalent to α · (# relocations) + β · crane_time.
    """
    return alpha * plan.num_moves() + beta * compute_crane_time(plan, kin)


# ================================================================ #
#  Plan analytics                                                    #
# ================================================================ #

def _relocation_counts(plan: RelocationPlan) -> Dict[int, int]:
    """Return {container_id: number of relocation moves}."""
    counts: Dict[int, int] = {}
    for m in plan.movements:
        if not m.is_retrieval:
            counts[m.container_id] = counts.get(m.container_id, 0) + 1
    return counts


def _last_relocation_index(
    plan:         RelocationPlan,
    container_id: int,
) -> Optional[int]:
    """Index of the container's LAST relocation; ``None`` if it was never
    relocated."""
    last = None
    for i, m in enumerate(plan.movements):
        if m.container_id == container_id and not m.is_retrieval:
            last = i
    return last


# ================================================================ #
#  Plan re-construction with RIL repair                              #
# ================================================================ #

def _replay_with_edit(
    initial_yard,
    containers:   List,
    plan:         RelocationPlan,
    edit_idx:     int,
    new_dst:      Tuple[int, int],
    max_tiers:    int,
) -> Optional[RelocationPlan]:
    """
    Re-simulate ``plan`` on a fresh copy of ``initial_yard``:

      • movements [0 .. edit_idx-1] are applied as-is;
      • movement at ``edit_idx`` is applied with ``to_pos = new_dst``;
      • everything after that point is discarded and the remaining
        retrievals are rebuilt greedily with the RIL heuristic.

    Returns the new plan, or ``None`` if the edit cannot be applied
    (source no longer has the expected top / destination already full
    / no RIL-feasible repair).
    """
    sim      = copy.deepcopy(initial_yard)
    new_plan = RelocationPlan()

    # ── 1. Replay unchanged prefix ─────────────────────────────── #
    for i in range(edit_idx):
        m   = plan.movements[i]
        stk = sim.stacks.get(m.from_pos)
        if stk is None or stk.is_empty or stk.top.id != m.container_id:
            return None
        if m.is_retrieval:
            stk.pop()
            new_plan.add(Movement(m.container_id, m.from_pos, None))
        else:
            dst_stk = sim.stacks.get(m.to_pos)
            if dst_stk is None or dst_stk.height >= max_tiers:
                return None
            sim.relocate(m.from_pos, m.to_pos)
            new_plan.add(Movement(m.container_id, m.from_pos, m.to_pos))

    # ── 2. Apply the edited move ───────────────────────────────── #
    edited = plan.movements[edit_idx]
    if edited.is_retrieval:
        return None   # editing a retrieval is meaningless for the search

    src_stk = sim.stacks.get(edited.from_pos)
    dst_stk = sim.stacks.get(new_dst)
    if src_stk is None or src_stk.is_empty:
        return None
    if src_stk.top.id != edited.container_id:
        return None
    if new_dst == edited.from_pos:
        return None
    if dst_stk is None or dst_stk.height >= max_tiers:
        return None

    sim.relocate(edited.from_pos, new_dst)
    new_plan.add(Movement(edited.container_id, edited.from_pos, new_dst))

    # ── 3. Figure out which priorities are still to be retrieved ─ #
    id_to_priority: Dict[int, int] = {c.id: c.priority for c in containers}
    retrieved_priorities: Set[int] = {
        id_to_priority[m.container_id]
        for m in new_plan.movements
        if m.is_retrieval and m.container_id in id_to_priority
    }
    priority_map: Dict[int, object] = {c.priority: c for c in containers}
    remaining = sorted(
        p for p in priority_map if p not in retrieved_priorities
    )

    # ── 4. Rebuild the remainder with RIL ──────────────────────── #
    for pri in remaining:
        target = priority_map[pri]
        stk    = sim._find_stack(target)
        if stk is None:
            return None
        src = (stk.bay, stk.row)

        while stk.top is not None and stk.top != target:
            blocker = stk.top
            dst = ril_select_dst(sim, blocker.priority, src, max_tiers)
            if dst is None:
                return None
            new_plan.add(Movement(blocker.id, src, dst))
            sim.relocate(src, dst)

        new_plan.add(Movement(target.id, src, None))
        stk.pop()

    return new_plan


# ================================================================ #
#  Hill-climbing driver                                              #
# ================================================================ #

def local_search(
    plan:          RelocationPlan,
    initial_yard,
    containers:    List,
    max_tiers:     int,
    kin:           KinematicsModel,
    alpha:         float,
    beta:          float,
    max_iter:      int,
) -> Tuple[RelocationPlan, float]:
    """
    One hill-climbing pass over ``plan`` (paper §4.4).  Returns the
    (possibly) improved plan together with its evaluation value.
    """
    best_plan = plan
    best_eval = eval_plan(best_plan, kin, alpha, beta)

    tried: Set[int] = set()

    all_positions = list(initial_yard.stacks.keys())

    for _ in range(max_iter):
        counts = _relocation_counts(best_plan)
        remaining_cids = [
            cid for cid, cnt in counts.items()
            if cnt > 0 and cid not in tried
        ]
        if not remaining_cids:
            break

        # Pick container most relocated (ties → smaller id for determinism)
        picked = max(remaining_cids, key=lambda cid: (counts[cid], -cid))

        edit_idx = _last_relocation_index(best_plan, picked)
        if edit_idx is None:
            tried.add(picked)
            continue

        edited_move = best_plan.movements[edit_idx]
        original_dst = edited_move.to_pos

        improved = False
        for cand in all_positions:
            if cand == edited_move.from_pos or cand == original_dst:
                continue
            candidate_plan = _replay_with_edit(
                initial_yard,
                containers,
                best_plan,
                edit_idx,
                cand,
                max_tiers,
            )
            if candidate_plan is None:
                continue
            cand_eval = eval_plan(candidate_plan, kin, alpha, beta)
            if cand_eval < best_eval:
                best_plan = candidate_plan
                best_eval = cand_eval
                tried.clear()
                improved = True
                break

        if not improved:
            tried.add(picked)

    return best_plan, best_eval
