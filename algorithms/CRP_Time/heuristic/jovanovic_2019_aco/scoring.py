"""
MinMax heuristic and pheromone helpers for JovanovicACO_CRPTime.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from core.objectives import ObjectiveSpec

# Truck pickup position — must match core/objectives.py _TRUCK_POS
_TRUCK_POS: Tuple[int, int] = (0, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Stack value helpers  (copied from CRP_R/heuristic/jovanovic_2019_aco/scoring.py)
# ─────────────────────────────────────────────────────────────────────────────

def dd(stack_priorities: List[int], n_total: int) -> int:
    """
    dd(S) — minimum due-date (most urgent container) in stack S.  Eq. 2.
    Empty stack → N + 1  (sentinel: no container, never blocks anything).
    """
    if not stack_priorities:
        return n_total + 1
    return min(stack_priorities)


def dd_star(stack_priorities: List[int], n_total: int, stack_1based_idx: int) -> int:
    """
    dd*(S) — extended dd for pheromone-matrix indexing.

    Non-empty stack: returns dd(S) = min due-date ∈ {1..N}.
    Empty stack    : returns N + stack_1based_idx ∈ {N+1 .. N+W}.
      This differentiates empty stacks so the pheromone matrix can learn
      which empty slot is most beneficial.
    """
    if not stack_priorities:
        return n_total + stack_1based_idx
    return min(stack_priorities)


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic scoring  (copied — unchanged for crane-time; per paper Section 6)
# ─────────────────────────────────────────────────────────────────────────────

def dif(c: int, d: int, n_total: int) -> int:
    """
    Eq. 3 — undesirability of placing container c above a stack whose
    minimum due-date is d.

    The MinMax heuristic function is kept identical for the crane-time
    variant (Section 6 of the paper uses the same heuristic; only val()
    and the lower bound change).

    Range: 1 .. 2N.  Smaller = more desirable destination.
    """
    if d > c:
        return d - c
    else:
        return 2 * n_total + 1 - d


# ─────────────────────────────────────────────────────────────────────────────
# Lower bound  (copied)
# ─────────────────────────────────────────────────────────────────────────────

def compute_lb(stacks: Dict[Any, List[int]]) -> int:
    """
    Lower bound on remaining relocations: number of non-well-located
    containers in the current bay state.

    Container c is non-well-located if ∃ d below c in the same stack with
    d < c  (d is retrieved before c but c is blocking d).
    """
    lb = 0
    for prios in stacks.values():
        for i in range(1, len(prios)):
            c = prios[i]
            if min(prios[:i]) < c:
                lb += 1
    return lb


# ─────────────────────────────────────────────────────────────────────────────
# Crane-time lower bounds LB_f and LB_v  (Sec. 6, eqs. 44–45, p. 85)
# ─────────────────────────────────────────────────────────────────────────────

def has_time_lb(spec: ObjectiveSpec) -> bool:
    """The paper defines a crane-time bound only for O_f (f2) and O_v (f2vert)."""
    return spec.mode == "crane_time" and spec.time_model in ("f2", "f2_vertical")


def lb_container(s: int, t: int, nw: bool, spec: ObjectiveSpec) -> float:
    """
    Contribution of one container at stack s, tier t to LB_f (eq. 44) or
    LB_v (eq. 45): its retrieval from the current position, plus the
    minimal relocation cost if it is non-well-located (nw).

    Taken literally from the paper; LB_f is not always a valid lower bound
    (see docs/validation/jovanovic_2019.md §3.6).
    """
    ts = spec.stack_s_per_stack
    if spec.time_model == "f2":
        tpp = spec.pickup_place_s
        return tpp + 2.0 * s * ts + ((tpp + ts) if nw else 0.0)
    tr = spec.empty_vertical_s_per_tier + spec.loaded_vertical_s_per_tier
    h_max = spec.max_tiers + 1
    return (
        (2.0 * h_max - t - spec.outside_height) * tr
        + 2.0 * s * ts
        + ((2.0 * tr + ts) if nw else 0.0)
    )


def lb_time(
    stacks: Dict[Any, List[int]],
    key_to_1based_idx: Dict[Any, int],
    spec: ObjectiveSpec,
) -> float:
    """LB_f or LB_v of a bay (tiers 1-based from the ground); 0 without a paper bound."""
    if not has_time_lb(spec):
        return 0.0
    total = 0.0
    for key, prios in stacks.items():
        below_min = None
        for i, c in enumerate(prios):
            nw = below_min is not None and below_min < c
            total += lb_container(key_to_1based_idx[key], i + 1, nw, spec)
            below_min = c if below_min is None else min(below_min, c)
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Inline 2-D kinematics  (new — adapts KinematicsModel to avoid object overhead)
# ─────────────────────────────────────────────────────────────────────────────

def _travel_time(
    pos_a:     Tuple[int, int],
    pos_b:     Tuple[int, int],
    gantry_s:  float,
    trolley_s: float,
    accel_s:   float,
) -> float:
    """
    Repositioning time from pos_a to pos_b for a 2-D RMGC.

    Gantry (bay axis) and trolley (row axis) move simultaneously, so total
    travel time = max(gantry_time, trolley_time).
    Acceleration overhead is added once when the gantry actually moves.

    Mirrors KinematicsModel.travel_time() from core/objectives.py.
    """
    bay_diff  = abs(pos_b[0] - pos_a[0])
    row_diff  = abs(pos_b[1] - pos_a[1])
    gantry_t  = bay_diff * gantry_s + (accel_s if bay_diff > 0 else 0.0)
    trolley_t = row_diff * trolley_s
    return max(gantry_t, trolley_t)


def move_time_inline(
    crane_pos:  Tuple[int, int],
    src_pos:    Tuple[int, int],
    dst_pos:    Tuple[int, int],
    gantry_s:   float,
    trolley_s:  float,
    accel_s:    float,
    spreader_s: float,
) -> Tuple[float, Tuple[int, int]]:
    """
    Crane time (seconds) to relocate a container from src_pos to dst_pos,
    given the crane is currently at crane_pos.

    Sequence (mirrors KinematicsModel.move_time()):
      1. Reposition crane: crane_pos → src_pos
      2. Pickup + carry:   src_pos   → dst_pos
      3. Place-down:       spreader_s (combined pickup+place-down)

    Returns
    -------
    cost          : total seconds for this move
    new_crane_pos : dst_pos  (crane ends at destination)
    """
    cost = (
        _travel_time(crane_pos, src_pos, gantry_s, trolley_s, accel_s)
        + _travel_time(src_pos, dst_pos, gantry_s, trolley_s, accel_s)
        + spreader_s
    )
    return cost, dst_pos


def retrieval_time_inline(
    crane_pos:  Tuple[int, int],
    src_pos:    Tuple[int, int],
    gantry_s:   float,
    trolley_s:  float,
    accel_s:    float,
    spreader_s: float,
) -> Tuple[float, Tuple[int, int]]:
    """
    Crane time (seconds) to retrieve a container from src_pos to the truck
    at _TRUCK_POS = (0, 0).

    Returns
    -------
    cost          : total seconds for this retrieval
    new_crane_pos : _TRUCK_POS  (crane ends at truck position)
    """
    cost = (
        _travel_time(crane_pos, src_pos, gantry_s, trolley_s, accel_s)
        + _travel_time(src_pos, _TRUCK_POS, gantry_s, trolley_s, accel_s)
        + spreader_s
    )
    return cost, _TRUCK_POS


# ─────────────────────────────────────────────────────────────────────────────
# Crane-time lower bound  (new — adapts LB_f Eq. 44 to platform 2D model)
# ─────────────────────────────────────────────────────────────────────────────

def compute_lb_time(n_nwl: int, spreader_s: float) -> float:
    """
    Lower bound on remaining crane time.

    Each of the n_nwl non-well-located containers requires at least one
    extra relocation costing at minimum spreader_s seconds (the irreducible
    pickup + place-down time when crane is already at source and source/
    destination share the same bay and row — i.e. travel time = 0).

    This corresponds to the second term of LB_f (Eq. 44) adapted to the
    platform's 2D kinematics: Σ_{c∈NW} (t_pp + t_s) ≥ n_nwl × spreader_s.

    Returns
    -------
    float  lower-bound crane time in seconds
    """
    return n_nwl * spreader_s


# ─────────────────────────────────────────────────────────────────────────────
# Greedy warm-start with crane-time accumulation  (new)
# ─────────────────────────────────────────────────────────────────────────────

def run_greedy_rbrp_time(
    stacks_init:       Dict[Any, List[int]],
    all_keys:          List[Any],
    n_total:           int,
    max_tiers:         int,
    key_to_1based_idx: Dict[Any, int],
    gantry_s:          float,
    trolley_s:         float,
    accel_s:           float,
    spreader_s:        float,
) -> Tuple[List[Tuple[int, int, int, int]], float]:
    """
    MinMax greedy rBRP warm-start that accumulates crane time instead of
    counting relocations.

    Stack keys are (bay, row) tuples, so they double as spatial positions
    for the kinematics model — no separate position mapping is needed.

    Initial crane position is (1, 1), matching compute_crane_time() default
    in core/objectives.py.

    Returns
    -------
    solution   : list of (c, d_star, mc, t) 4-tuples for pheromone seeding
    crane_time : total crane time in seconds for this greedy solution
    """
    stacks: Dict[Any, List[int]] = {k: list(v) for k, v in stacks_init.items()}
    M: Dict[int, int] = {}
    solution: List[Tuple[int, int, int, int]] = []
    crane_pos: Tuple[int, int] = (1, 1)
    total_time: float = 0.0

    for target in range(1, n_total + 1):
        src_key: Optional[Any] = next(
            (k for k, p in stacks.items() if target in p), None
        )
        if src_key is None:
            continue

        # Relocate all blockers above target (top-first), MinMax destination
        while stacks[src_key] and stacks[src_key][-1] != target:
            c = stacks[src_key][-1]

            best_dst: Optional[Any] = None
            best_score = float("inf")
            for dst_key in all_keys:
                if dst_key == src_key:
                    continue
                if len(stacks[dst_key]) >= max_tiers:
                    continue
                score = dif(c, dd(stacks[dst_key], n_total), n_total)
                if score < best_score:
                    best_score = score
                    best_dst   = dst_key

            # R_c is empty (eq. 1): Alg. 1 cannot continue, so S_g and
            # tau_0 (eq. 37) are undefined.
            if best_dst is None:
                raise ValueError(
                    f"Jovanović greedy start (Alg. 1): no free stack for container "
                    f"{c} above target {target}; every other stack is full"
                )

            d_val = dd_star(stacks[best_dst], n_total, key_to_1based_idx[best_dst])
            mc    = M.get(c, 0)
            solution.append((c, d_val, mc, target))

            # Accumulate crane time for relocation; src_key IS src_pos (bay, row)
            cost, crane_pos = move_time_inline(
                crane_pos, src_key, best_dst, gantry_s, trolley_s, accel_s, spreader_s
            )
            total_time += cost

            stacks[src_key].pop()
            stacks[best_dst].append(c)
            M[c] = mc + 1

        # Retrieve target (also incurs crane time)
        if stacks[src_key] and stacks[src_key][-1] == target:
            cost, crane_pos = retrieval_time_inline(
                crane_pos, src_key, gantry_s, trolley_s, accel_s, spreader_s
            )
            total_time += cost
            stacks[src_key].pop()

    return solution, total_time
