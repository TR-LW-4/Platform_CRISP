"""
Phase-3 crane-time waypoints for LeeLeeRetrievalHeuristic.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import multiprocessing as mp
import random
from typing import List, Optional, Tuple

from core.plan import RelocationPlan, simulate_plan
from core.objectives import KinematicsModel, compute_crane_time
from .phase2 import build_2hop_plan, swap_intermediate


def phase3_reduce_time(
    plan:           RelocationPlan,
    initial_yard,
    max_tiers:      int,
    kinematics:     KinematicsModel,
    max_no_improve: int = 500,
    stop_event:     Optional[mp.Event] = None,
) -> Tuple[RelocationPlan, List[float]]:
    """
    Reduce crane working time by trying alternate waypoints for Type B
    containers, without increasing the total number of moves.

    Returns
    -------
    (best_plan, crane_time_history)
    """
    best      = plan.clone()
    best_time = compute_crane_time(best, kinematics)

    all_positions = list(initial_yard.stacks.keys())
    history       = [best_time]
    no_improve    = 0

    while no_improve < max_no_improve:
        if stop_event is not None and stop_event.is_set():
            break

        type_b_ids = best.type_b_container_ids()
        if not type_b_ids:
            break

        cid       = random.choice(type_b_ids)
        cid_moves = best.moves_of(cid)

        if len(cid_moves) < 2:
            no_improve += 1
            continue

        current_tos = {m.to_pos for _, m in cid_moves}
        initial_src = cid_moves[0][1].from_pos

        shuffled = all_positions[:]
        random.shuffle(shuffled)
        improved = False

        for new_pos in shuffled:
            if new_pos in current_tos or new_pos == initial_src:
                continue

            # Shorten to 2-hop if still multi-hop; otherwise swap the waypoint
            if len(cid_moves) >= 3:
                candidate = build_2hop_plan(best, cid, new_pos)
            else:
                candidate = swap_intermediate(best, cid, new_pos)

            if candidate is None:
                continue

            # Phase 3 constraint: must not increase total moves
            if candidate.num_moves() > best.num_moves():
                continue

            result = simulate_plan(initial_yard, candidate, max_tiers)
            if not result.feasible:
                continue

            cand_time = compute_crane_time(candidate, kinematics)
            if cand_time < best_time:
                best      = candidate
                best_time = cand_time
                history.append(best_time)
                no_improve = 0
                improved   = True
                break

        if not improved:
            no_improve += 1

    return best, history
