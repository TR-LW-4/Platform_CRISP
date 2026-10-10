"""
One ant of JovanovicACO_CRPTime: a complete retrieval of the bay under the
restricted rule, where every blocker of the current target goes to a stack chosen
by the transition rule (Sec. 5, p. 82–83), and the ant stops as soon as its crane
time plus the lower bound of the remaining bay reaches the best cost (Alg. 2).
Also the decoding of a solution into moves.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from algorithms.CRP_Time.utils.crane_time import move_time
from algorithms.CRP_Time.utils.located import min_due, non_located
from core.objectives import ObjectiveSpec
from core.plan import Movement

from .lower_bound import lb_container, lb_time
from .minmax import dif
from .pheromone import Solution


@dataclass(frozen=True)
class Start:
    """The initial bay every ant starts from, and what follows from it. Stack indices are 0-based;
    stack s of the paper is index s - 1."""

    stacks:    List[List[int]]
    positions: List[Tuple[int, int]]
    n_total:   int
    max_tiers: int
    loc:       List[Optional[int]]   # stack of container c at index c
    smin:      List[int]             # minimum due date per stack, N + 1 if empty
    nw:        List[bool]            # non-well-located per container at index c
    lb:        float                 # LB(Bay) of the initial bay, 0 for the relocation objective


def move_cost(move: Movement, spec: ObjectiveSpec) -> float:
    """Cost of one move under the objective: its crane time, or 1 per relocation under the relocation
    objective of the paper (kept for the validation in the paper's own setting)."""
    if spec.mode == "relocations":
        return 0.0 if move.is_retrieval else 1.0
    return move_time(move, spec)


def start(stacks: List[List[int]], positions: List[Tuple[int, int]], max_tiers: int,
          spec: ObjectiveSpec) -> Start:
    n_total = sum(len(st) for st in stacks)
    loc: List[Optional[int]] = [None] * (n_total + 1)
    nw = [False] * (n_total + 1)
    for s, stack in enumerate(stacks):
        for c, bad in zip(stack, non_located(stack)):
            loc[c] = s
            nw[c] = bad
    return Start(stacks, positions, n_total, max_tiers, loc,
                 [min_due(stack, n_total + 1) for stack in stacks], nw,
                 lb_time(stacks, spec) if spec.mode == "crane_time" else 0.0)


def build_solution(
    start:     Start,
    pheromone: np.ndarray,
    q0:        float,
    max_moves: int,
    best_cost: float,
    spec:      ObjectiveSpec,
) -> Tuple[Solution, bool, float]:
    """Returns the ant's solution, whether it is complete (not aborted and never blocked), and the cost
    of its moves so far."""
    n_total, max_tiers, positions = start.n_total, start.max_tiers, start.positions
    n_stacks = len(start.stacks)
    stacks       = [list(st) for st in start.stacks]
    M            = [0] * (n_total + 1)
    S: Solution  = []
    valid        = True
    loc          = list(start.loc)
    smin         = list(start.smin)
    time_so_far  = 0.0               # cost of the moves so far
    nw           = list(start.nw)
    lb_cur       = start.lb          # LB(Bay) of the current bay
    use_lb       = spec.mode == "crane_time"

    for target in range(1, n_total + 1):
        if not valid:
            break

        src = loc[target]
        if src is None:
            continue

        # ── Relocate all blockers above target ────────── #
        while valid and stacks[src] and stacks[src][-1] != target:
            c = stacks[src][-1]

            # Candidate destinations with g = heuristic value * pheromone
            best_g   = -1.0
            best_dst = -1
            best_d   = 0
            c_i      = c - 1
            mc_cl    = min(M[c], max_moves)
            t_i      = target - 1
            cands_g: List[float] = []
            cands_k: List[int]   = []
            cands_d: List[int]   = []

            for dst in range(n_stacks):
                if dst == src:
                    continue
                if len(stacks[dst]) >= max_tiers:
                    continue

                # dd* from the running stack minimum
                sm    = smin[dst]
                d_val = sm if sm <= n_total else n_total + dst + 1

                f_val = 1.0 / (1.0 + dif(c, sm, n_total))
                tau   = float(pheromone[c_i, d_val - 1, mc_cl, t_i])
                g     = f_val * tau
                cands_g.append(g)
                cands_k.append(dst)
                cands_d.append(d_val)

                if g > best_g:
                    best_g   = g
                    best_dst = dst
                    best_d   = d_val

            if not cands_k:
                valid = False
                break

            # Transition rule: exploit with probability q0, otherwise roulette over g
            if np.random.uniform() < q0 or len(cands_k) == 1:
                dst   = best_dst
                d_val = best_d
            else:
                g_arr = np.asarray(cands_g, dtype=np.float64)
                g_sum = g_arr.sum()
                probs = (
                    g_arr / g_sum
                    if g_sum > 0.0
                    else np.ones(len(cands_k)) / len(cands_k)
                )
                idx   = np.random.choice(len(cands_k), p=probs)
                dst   = cands_k[idx]
                d_val = cands_d[idx]

            # Record the 4-tuple and apply the relocation
            S.append((c, d_val, mc_cl, target))
            movement = Movement(
                c,
                positions[src],
                positions[dst],
                from_tier=len(stacks[src]),
                to_tier=len(stacks[dst]) + 1,
            )
            nw_new = smin[dst] < c

            stacks[src].pop()
            if smin[src] == c:
                smin[src] = (
                    min(stacks[src])
                    if stacks[src]
                    else (n_total + 1)
                )

            stacks[dst].append(c)
            if c < smin[dst]:
                smin[dst] = c

            loc[c]  = dst
            M[c]   += 1

            time_so_far += move_cost(movement, spec)

            if use_lb:
                lb_cur += lb_container(
                    dst + 1, movement.to_tier, nw_new, spec,
                ) - lb_container(
                    src + 1, movement.from_tier, nw[c], spec,
                )
                nw[c] = nw_new

            # Alg. 2: stop once the partial cost plus LB(Bay)
            # reaches the best cost.
            if time_so_far + lb_cur >= best_cost:
                valid = False

        # Retrieve target (also incurs crane time)
        if (
            valid
            and stacks[src]
            and stacks[src][-1] == target
        ):
            movement = Movement(
                target,
                positions[src],
                None,
                from_tier=len(stacks[src]),
            )
            time_so_far += move_cost(movement, spec)
            # The retrieval cost equals the term it removes
            # from LB(Bay), so no abort check is needed here.
            if use_lb:
                lb_cur -= lb_container(
                    src + 1, movement.from_tier, nw[target], spec,
                )

            stacks[src].pop()
            if smin[src] == target:
                smin[src] = (
                    min(stacks[src])
                    if stacks[src]
                    else (n_total + 1)
                )
            loc[target] = None

    return S, valid, time_so_far


def decode(solution: Solution, stacks: List[List[int]], n_total: int) -> List[Tuple[int, Optional[int]]]:
    """Moves (src, dst) on stack indices for a solution, dst None for a retrieval: per target, its
    relocations in order, then its retrieval."""
    loc: Dict[int, Any] = {c: s for s, stack in enumerate(stacks) for c in stack}
    by_target: Dict[int, List[Tuple[int, int, int, int]]] = {}
    for record in solution:
        by_target.setdefault(record[3], []).append(record)

    moves: List[Tuple[int, Optional[int]]] = []
    for target in range(1, n_total + 1):
        for c, d_val, _mc, _t in by_target.get(target, []):
            dst = loc[d_val] if d_val <= n_total else d_val - n_total - 1
            moves.append((loc[c], dst))
            loc[c] = dst
        moves.append((loc[target], None))
        loc[target] = None
    return moves
