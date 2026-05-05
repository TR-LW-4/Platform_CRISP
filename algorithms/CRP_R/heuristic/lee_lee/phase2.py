"""
Lee & Lee (2010) – Phase 2: Move reduction.

Iteratively finds Type B containers (relocated ≥1 time) and tries to
replace their multi-hop path with a shorter 2-hop path:
  (c: initial → new_waypoint)  then  (c: new_waypoint → OUT).

Accepts any improvement that reduces total moves and passes feasibility
simulation.  Terminates when the lower bound is reached or
`max_no_improve` consecutive iterations fail to improve.
"""

from __future__ import annotations

import copy
import multiprocessing as mp
import random
from typing import List, Optional, Tuple

from core.plan import Movement, RelocationPlan, simulate_plan
from core.objectives import lower_bound_relocations


def build_2hop_plan(
    plan:             RelocationPlan,
    container_id:     int,
    new_intermediate: Tuple[int, int],
) -> Optional[RelocationPlan]:
    """
    Replace container_id's multi-hop path with a 2-hop path via new_intermediate.

    Returns None if the container already has ≤ 2 moves.
    """
    cid_moves = plan.moves_of(container_id)
    if len(cid_moves) < 3:
        return None

    first_idx, first_move = cid_moves[0]
    last_idx,  _          = cid_moves[-1]
    skip_indices = {idx for idx, _ in cid_moves[1:-1]}

    new_movements = []
    for i, m in enumerate(plan.movements):
        if i == first_idx:
            new_movements.append(Movement(
                container_id=container_id,
                from_pos=first_move.from_pos,
                to_pos=new_intermediate,
            ))
        elif i == last_idx:
            new_movements.append(Movement(
                container_id=container_id,
                from_pos=new_intermediate,
                to_pos=None,
            ))
        elif i in skip_indices:
            pass  # drop intermediate moves
        else:
            new_movements.append(copy.copy(m))

    return RelocationPlan(movements=new_movements)


def swap_intermediate(
    plan:             RelocationPlan,
    container_id:     int,
    new_intermediate: Tuple[int, int],
) -> Optional[RelocationPlan]:
    """
    For a container already on a 2-hop path, try a different waypoint.
    Used by Phase 3 to optimise crane time without changing move count.
    """
    cid_moves = plan.moves_of(container_id)
    if len(cid_moves) != 2:
        return None

    first_idx, first_move = cid_moves[0]
    last_idx,  _          = cid_moves[1]

    new_movements = []
    for i, m in enumerate(plan.movements):
        if i == first_idx:
            new_movements.append(Movement(
                container_id=container_id,
                from_pos=first_move.from_pos,
                to_pos=new_intermediate,
            ))
        elif i == last_idx:
            new_movements.append(Movement(
                container_id=container_id,
                from_pos=new_intermediate,
                to_pos=None,
            ))
        else:
            new_movements.append(copy.copy(m))

    return RelocationPlan(movements=new_movements)


def phase2_reduce_moves(
    plan:           RelocationPlan,
    initial_yard,
    containers:     list,
    max_tiers:      int,
    max_no_improve: int = 200,
    stop_event:     Optional[mp.Event] = None,
) -> Tuple[RelocationPlan, List[int]]:
    """
    Reduce total moves by shortening Type B container paths to 2 hops.

    Returns
    -------
    (best_plan, move_count_history)
    """
    best         = plan.clone()
    n_containers = len(containers)
    lb_total     = n_containers + lower_bound_relocations(initial_yard)
    all_positions = list(initial_yard.stacks.keys())

    history    = [best.num_moves()]
    no_improve = 0

    while no_improve < max_no_improve:
        if stop_event is not None and stop_event.is_set():
            break
        if best.num_moves() <= lb_total:
            break

        type_b_ids = best.type_b_container_ids()
        if not type_b_ids:
            break

        cid       = random.choice(type_b_ids)
        cid_moves = best.moves_of(cid)

        if len(cid_moves) < 3:
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

            candidate = build_2hop_plan(best, cid, new_pos)
            if candidate is None:
                continue

            result = simulate_plan(initial_yard, candidate, max_tiers)
            if result.feasible and candidate.num_moves() < best.num_moves():
                best = candidate
                history.append(best.num_moves())
                no_improve = 0
                improved   = True
                break

        if not improved:
            no_improve += 1

    return best, history
