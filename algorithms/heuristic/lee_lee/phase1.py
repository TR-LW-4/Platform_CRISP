"""
Lee & Lee (2010) – Phase 1: Greedy initial feasible sequence.

Retrieves containers in priority order 1…N.
When the target is buried, relocates all blockers to the nearest
available stack (Manhattan distance), greedily.
"""

from __future__ import annotations

import copy
from typing import List, Optional, Tuple

from core.plan import Movement, RelocationPlan


def nearest_dst(
    yard,
    src_pos: Tuple[int, int],
) -> Optional[Tuple[int, int]]:
    """
    Find the closest non-full stack to src_pos (Manhattan distance).
    Ties broken lexicographically by (bay, row).
    Returns None if all stacks are full.
    """
    best: Optional[Tuple[int, int]] = None
    best_dist = 10 ** 9
    src_bay, src_row = src_pos

    for (bay, row), stack in yard.stacks.items():
        if (bay, row) == src_pos or stack.is_full:
            continue
        dist = abs(bay - src_bay) + abs(row - src_row)
        if dist < best_dist or (dist == best_dist and best is not None
                                and (bay, row) < best):
            best_dist = dist
            best = (bay, row)

    return best


def phase1_greedy(yard, containers: list, num_containers: int) -> RelocationPlan:
    """
    Generate an initial feasible plan by retrieving containers in priority order.

    Parameters
    ----------
    yard           : initial Yard state (deep-copied internally, not mutated)
    containers     : list of Container objects for the episode
    num_containers : total number of containers (= max priority value)

    Returns
    -------
    RelocationPlan with all containers retrieved (may have many relocations)
    """
    sim  = copy.deepcopy(yard)
    plan = RelocationPlan()

    priority_map = {c.priority: c for c in containers}

    for priority in range(1, num_containers + 1):
        target = priority_map.get(priority)
        if target is None:
            continue

        target_stack = sim._find_stack(target)
        if target_stack is None:
            continue

        src_pos = (target_stack.bay, target_stack.row)

        # Relocate all blockers above target, top-to-bottom
        while target_stack.top is not None and target_stack.top != target:
            blocker = target_stack.top
            dst = nearest_dst(sim, src_pos)
            if dst is None:
                raise RuntimeError("Yard is full – cannot relocate blocker")

            plan.add(Movement(
                container_id=blocker.id,
                from_pos=src_pos,
                to_pos=dst,
            ))
            sim.relocate(src_pos, dst)

        # Retrieve target
        plan.add(Movement(
            container_id=target.id,
            from_pos=src_pos,
            to_pos=None,
        ))
        target_stack.pop()

    return plan
