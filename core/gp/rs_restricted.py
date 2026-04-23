"""
Restricted Relocation Scheme (paper §5, Algorithm 2) driven by a
GP-evolved Priority Function (PF).

The RS iterates over the fixed retrieval order 1 → N.  For each target:

1. Find the container's current stack.
2. While the target is not on top, score every admissible destination
   stack with the PF tree, move the topmost blocker to the argmin-score
   stack.
3. Retrieve the target.

Under the restricted regime, destinations other than the source are
always admissible (non-full); no proactive cleaning of other stacks is
performed.
"""

from __future__ import annotations

import copy
from typing import Callable, Dict, List

from core.objectives import KinematicsModel
from core.plan       import Movement, RelocationPlan

from .engine import Node


def build_plan_restricted(
    yard,
    containers:     List,
    pf_tree:        Node,
    terminal_table: Dict[str, Callable[[dict], float]],
    max_tiers:      int,
    kin:            KinematicsModel,
) -> RelocationPlan:
    """
    Build a complete ``RelocationPlan`` using the restricted RS
    coupled with the given PF tree.

    The tree evaluates to a scalar priority for each candidate
    destination; the RS always picks the destination with the
    SMALLEST score (paper convention).
    """
    sim  = copy.deepcopy(yard)
    plan = RelocationPlan()

    priority_map: Dict[int, object] = {c.priority: c for c in containers}

    for pri in sorted(priority_map.keys()):
        target = priority_map[pri]
        stk    = sim._find_stack(target)
        if stk is None:
            continue
        src = (stk.bay, stk.row)

        while stk.top is not None and stk.top != target:
            blocker = stk.top

            best_dst   = None
            best_score = float("inf")

            for pos, dst_stk in sim.stacks.items():
                if pos == src or dst_stk.height >= max_tiers:
                    continue

                ctx = {
                    "sim":       sim,
                    "src":       src,
                    "dst":       pos,
                    "blocker":   blocker,
                    "kin":       kin,
                    "max_tiers": max_tiers,
                }
                # Evaluate every terminal value used by the tree lazily via ctx;
                # the GP Node.evaluate() reads terminals by name from ctx, so
                # we expose pre-computed values here to avoid recomputation.
                for tname, tfn in terminal_table.items():
                    ctx[tname] = tfn(ctx)

                score = pf_tree.evaluate(ctx)
                # Deterministic tie-break (smaller (bay, row) first)
                key   = (score, pos[0], pos[1])
                if (score < best_score) or (
                    score == best_score and best_dst is not None
                    and key < (best_score, best_dst[0], best_dst[1])
                ):
                    best_score = score
                    best_dst   = pos

            if best_dst is None:
                raise RuntimeError(
                    "Ðurasević-2024 restricted RS: no feasible destination "
                    f"for blocker C{blocker.id} at {src} (yard full)."
                )

            plan.add(Movement(
                container_id=blocker.id,
                from_pos=src,
                to_pos=best_dst,
            ))
            sim.relocate(src, best_dst)

        # Target now on top — retrieve it
        plan.add(Movement(
            container_id=target.id,
            from_pos=src,
            to_pos=None,
        ))
        stk.pop()

    return plan
