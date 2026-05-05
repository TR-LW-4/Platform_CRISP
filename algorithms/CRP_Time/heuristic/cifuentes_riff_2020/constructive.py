"""
G-CREM (Cifuentes & Riff, 2020) — Constructive phase.

Reference
---------
C.D. Cifuentes and M.C. Riff,
"G-CREM: A GRASP approach to solve the container relocation problem
for multibays", Applied Soft Computing 97 (2020) 106721.

Algorithm (paper Alg. 3)
------------------------
Iterate over the fixed retrieval order (priority 1 → N).  For each
target container:

  • If the target is on top of its stack, retrieve it immediately.
  • Otherwise, for the topmost blocker above the target:
        – Compute the myopic value  Mf_j = H − N_j  (= remaining capacity)
          for every candidate destination stack j (non-source, non-full).
        – Build the Restricted Candidate List: keep all stacks whose
          Mf belongs to the ``k`` best DISTINCT Mf values (RCL size is
          thus variable — every tie stays).
        – Randomly pick one destination from the RCL.
        – Relocate the blocker.
  • Repeat until the target is accessible, then retrieve it.

The resulting ``RelocationPlan`` is a feasible sequence of relocations
and retrievals that empties the yard.
"""

from __future__ import annotations

import copy
import random
from typing import Dict, List, Optional, Tuple

from core.plan import Movement, RelocationPlan


def _myopic_values(
    sim,
    src_pos:   Tuple[int, int],
    max_tiers: int,
) -> Dict[Tuple[int, int], int]:
    """
    Compute  Mf_j = H − height(j)  for every non-source, non-full stack.
    """
    mf: Dict[Tuple[int, int], int] = {}
    for pos, stk in sim.stacks.items():
        if pos == src_pos:
            continue
        avail = max_tiers - stk.height
        if avail <= 0:
            continue
        mf[pos] = avail
    return mf


def _build_rcl(
    mf: Dict[Tuple[int, int], int],
    k:  int,
) -> List[Tuple[int, int]]:
    """
    Keep every stack whose Mf matches one of the top-``k`` DISTINCT values.
    Sorted lexicographically on (bay, row) inside the list for determinism
    (the randomness comes from the caller's ``rng.choice``).
    """
    if not mf:
        return []
    distinct_sorted = sorted(set(mf.values()), reverse=True)
    top_values      = set(distinct_sorted[: max(1, k)])
    return sorted(pos for pos, val in mf.items() if val in top_values)


def constructive_phase(
    yard,
    containers:     List,
    k:              int,
    max_tiers:      int,
    rng:            random.Random,
) -> RelocationPlan:
    """
    Build one candidate ``RelocationPlan`` for ``yard`` with the G-CREM
    constructive procedure.

    Parameters
    ----------
    yard       : initial Yard state (deep-copied internally)
    containers : list of Container objects for this episode
    k          : number of distinct myopic values kept in the RCL
    max_tiers  : yard capacity per stack
    rng        : seeded ``random.Random`` instance for reproducibility
    """
    sim  = copy.deepcopy(yard)
    plan = RelocationPlan()

    priority_map: Dict[int, object] = {c.priority: c for c in containers}
    priorities   = sorted(priority_map.keys())

    for pri in priorities:
        target = priority_map[pri]
        stk    = sim._find_stack(target)
        if stk is None:
            continue

        src = (stk.bay, stk.row)

        # Relocate topmost blockers until the target is accessible
        while stk.top is not None and stk.top != target:
            blocker = stk.top
            mf      = _myopic_values(sim, src, max_tiers)
            rcl     = _build_rcl(mf, k)
            if not rcl:
                raise RuntimeError(
                    "G-CREM constructive: no feasible destination for "
                    f"blocker C{blocker.id} at {src} (yard is full)"
                )
            dst = rng.choice(rcl)
            plan.add(Movement(
                container_id=blocker.id,
                from_pos=src,
                to_pos=dst,
            ))
            sim.relocate(src, dst)

        # Target is now on top — retrieve it
        plan.add(Movement(
            container_id=target.id,
            from_pos=src,
            to_pos=None,
        ))
        stk.pop()

    return plan
