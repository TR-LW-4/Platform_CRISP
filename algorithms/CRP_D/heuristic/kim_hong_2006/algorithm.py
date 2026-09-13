"""
KimHong2006GroupENAR
<2006> <heuristic> <grouped> <single-bay> <CRP-D>
Group-priority ENAR relocation rule

------------------------------- Reference --------------------------------
K.H. Kim, G.-P. Hong,
"A heuristic rule for relocating blocks",
Computers & Operations Research 33 (2006) 940–954.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
import multiprocessing as mp
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig

from .enar import compute_enar_bay, enar_after_place


# ================================================================ #
#  Env helpers                                                      #
# ================================================================ #

def _n_stacks(env) -> int:
    return env.config.num_bays * env.config.num_rows


def _target_group(env) -> Optional[int]:
    return env._current_target_group()


def _stack_key(env, idx: int) -> Tuple[int, int]:
    return env._idx_to_stack(idx)


def _stacks_as_groups(env) -> List[List[int]]:
    """Return list of group-code lists (bottom→top) for each stack, in index order."""
    n = _n_stacks(env)
    result = []
    for i in range(n):
        key = _stack_key(env, i)
        stk = env.yard.stacks.get(key)
        if stk is None or stk.is_empty:
            result.append([])
        else:
            result.append([c.group for c in stk.containers])
    return result


def _max_tiers(env) -> int:
    return int(env.config.max_tiers)


# ================================================================ #
#  R[a] scoring helpers                                            #
# ================================================================ #

def _r_blocker(
    stacks_groups: List[List[int]],
    src_idx: int,
    dst_idx: int,
    max_tiers: int,
    current_enar: List[float],
) -> Tuple[float, List[List[int]]]:
    """
    Compute the marginal cost of moving the top of stack src_idx to dst_idx.

    Returns (delta_enar, new_stacks_groups).
    delta_enar = ENAR(dst after) - ENAR(dst before) + 0/1 (confirmed relocation
    cost is folded into the final r[a] count, not per-move here).
    """
    placed_group = stacks_groups[src_idx][-1] if stacks_groups[src_idx] else None
    if placed_group is None:
        return float("inf"), stacks_groups

    new_stacks = [list(s) for s in stacks_groups]
    new_stacks[src_idx] = new_stacks[src_idx][:-1]
    new_stacks[dst_idx] = new_stacks[dst_idx] + [placed_group]

    new_enar = compute_enar_bay(new_stacks, max_tiers)
    # Only dst changes in ENAR sum (src's ENAR change is same for all dst choices)
    delta = new_enar[dst_idx] - current_enar[dst_idx]
    return delta, new_stacks


def _best_dst_for_top(
    stacks_groups: List[List[int]],
    src_idx: int,
    max_tiers: int,
    current_enar: List[float],
    blocked_group: int,  # kept for signature compatibility; not used directly
) -> Tuple[int, float, List[List[int]]]:
    """
    Choose the best destination for the top of stack src_idx.

    Minimises delta_ENAR + confirmed_reloc:
      - confirmed_reloc = 1 if placed_group < max(dst stack groups after placing)
        (i.e. the relocated block would become badly placed), else 0.

    Returns (best_dst_idx, best_delta, new_stacks_groups).
    """
    n = len(stacks_groups)
    best_dst   = -1
    best_score = float("inf")
    best_new   = stacks_groups

    placed_group = stacks_groups[src_idx][-1] if stacks_groups[src_idx] else None
    if placed_group is None:
        return -1, float("inf"), stacks_groups

    for dst_idx in range(n):
        if dst_idx == src_idx:
            continue
        dst_stk = stacks_groups[dst_idx]
        if len(dst_stk) >= max_tiers:
            continue

        delta, new_stacks = _r_blocker(
            stacks_groups, src_idx, dst_idx, max_tiers, current_enar
        )
        # confirmed relocation cost: 1 if placed block has higher group code than
        # the destination stack's minimum (would be retrieved later than some in dst)
        dst_groups_after = new_stacks[dst_idx]
        dst_min = min(dst_groups_after) if dst_groups_after else 0
        confirmed = 1 if placed_group > dst_min else 0

        score = delta + confirmed
        if score < best_score:
            best_score = score
            best_dst   = dst_idx
            best_new   = new_stacks

    return best_dst, best_score, best_new


# ================================================================ #
#  Per-candidate planning (§3.3.2)                                #
# ================================================================ #

def _plan_for_target(
    env,
    tgt_stack_idx: int,
    tgt_depth: int,
    stacks_groups: List[List[int]],
    max_tiers: int,
) -> Tuple[int, int, List[Tuple[int, int]]]:
    """
    For a target container at `tgt_depth` from bottom in `tgt_stack_idx`,
    greedily assign destinations for each blocker top-down (§3.3.2).

    Returns:
      (total_R, n_realized_relocs, plan)
      plan = list of (src_idx, dst_idx) to execute top-down
    """
    n_stack = len(stacks_groups)
    stk     = stacks_groups[tgt_stack_idx]
    h       = len(stk)

    # Blockers are containers above tgt_depth (indices tgt_depth+1 .. h-1)
    n_blockers = h - 1 - tgt_depth   # number of containers above target

    if n_blockers <= 0:
        return 0, 0, []

    working = [list(s) for s in stacks_groups]
    plan: List[Tuple[int, int]] = []
    total_delta_enar: float = 0.0
    realized: int = 0

    for _ in range(n_blockers):
        current_enar = compute_enar_bay(working, max_tiers)
        src = tgt_stack_idx
        placed_group = working[src][-1] if working[src] else None
        if placed_group is None:
            break

        dst, delta, working = _best_dst_for_top(
            working, src, max_tiers, current_enar, placed_group
        )
        if dst < 0:
            break

        plan.append((src, dst))
        total_delta_enar += delta
        realized += 1

    # R[a] = delta_ENAR_sum + realized_relocations + confirmed_relocations_from_plan
    # Confirmed: after executing plan, how many containers in the bay block
    # a lower-group container in their new stack.
    # Simplified: r[a] = realized relocs (each blocker moved once).
    # The ENAR already accounts for future cost.
    total_R = int(round(total_delta_enar)) + realized
    return total_R, realized, plan


# ================================================================ #
#  Main per-step decision function                                 #
# ================================================================ #

def select_action(env) -> int:
    """
    Kim & Hong (2006) §3.3.2 group-priority heuristic.

    Returns the flat action integer for CRP-D (src_idx * S + dst_idx).
    """
    tg = _target_group(env)
    if tg is None:
        return 0

    n          = _n_stacks(env)
    max_t      = _max_tiers(env)
    stacks_grp = _stacks_as_groups(env)

    # ── Build list of accessible candidates ──────────────────────── #
    # A candidate is a stack that has at least one container of group tg.
    # Within that stack, consider the HIGHEST container of group tg
    # (a lower-tier target cannot be reached before the one above it).
    candidates: List[Tuple[int, int]] = []   # (stack_idx, depth_from_bottom)
    for s_idx in range(n):
        stk = stacks_grp[s_idx]
        for depth, g in enumerate(stk):
            if g == tg:
                # Take the topmost occurrence of tg in this stack
                # (last element with group == tg going bottom→top)
                pass   # we'll find it below
        # topmost tg container
        tg_depth = None
        for depth in range(len(stk) - 1, -1, -1):
            if stk[depth] == tg:
                tg_depth = depth
                break
        if tg_depth is None:
            continue
        # Only consider if it's reachable (it's the top, or all above can be moved)
        # In the CRP-D unrestricted model, we can always move the top first.
        candidates.append((s_idx, tg_depth))

    if not candidates:
        # Fallback: move topmost accessible container (should not happen in normal play)
        for s_idx in range(n):
            stk = env.yard.stacks.get(_stack_key(env, s_idx))
            if stk and not stk.is_empty:
                for dst_idx in range(n):
                    if dst_idx != s_idx:
                        dst_stk = env.yard.stacks.get(_stack_key(env, dst_idx))
                        if dst_stk and not dst_stk.is_full:
                            return s_idx * n + dst_idx
        return 0

    # ── For each candidate, compute plan and R[a] ─────────────────── #
    best_r        = float("inf")
    best_plan: List[Tuple[int, int]] = []

    for s_idx, tg_depth in candidates:
        total_r, realized, plan = _plan_for_target(
            env, s_idx, tg_depth, stacks_grp, max_t
        )
        if total_r < best_r:
            best_r    = total_r
            best_plan = plan

    # ── Execute the first step of the best candidate's plan ───────── #
    if best_plan:
        src_idx, dst_idx = best_plan[0]
        return src_idx * n + dst_idx

    # No blockers needed: target is already on top → environment handles retrieval
    # Find any candidate whose target is already on top
    for s_idx, tg_depth in candidates:
        stk = stacks_grp[s_idx]
        if tg_depth == len(stk) - 1:
            # Target is the topmost → the env auto-retrieves; no action needed
            # Return a dummy self-move that the env will treat as no-op (or use 0)
            return 0

    return 0


# ================================================================ #
#  BaseAlgorithm wrapper                                           #
# ================================================================ #

def _run_episode(env, selector_fn: Callable) -> Tuple[List[int], Dict]:
    env.reset()
    solution: List[int] = []
    done = False
    n    = _n_stacks(env)
    max_steps = n * getattr(env.config, "max_tiers", 10) * 4 + 200

    while not done and len(solution) < max_steps:
        mask = env._build_action_mask() if hasattr(env, "_build_action_mask") else None
        if mask is not None and not mask.any():
            break
        action = selector_fn(env)
        _, _, terminated, truncated, _ = env.step(action)
        solution.append(action)
        done = terminated or truncated

    return solution, env.get_metrics()


class KimHong2006GroupENAR(BaseAlgorithm):
    """
    Kim & Hong (2006) group-priority ENAR heuristic for CRP-D.

    Implements §3.2 (Equation 2) + §3.3.2 of:
      K.H. Kim and G.-P. Hong, "A heuristic rule for relocating blocks",
      Computers & Operations Research 33 (2006) 940–954.

    At each step the algorithm:
      1. Identifies all accessible target-group containers (candidates).
      2. For each candidate, greedily plans blocker relocations top-down,
         scoring each move with ENAR delta (Eq. 2) + realized relocations.
      3. Chooses the candidate whose total plan cost R[a] is minimum.
      4. Executes the first relocation of the winning plan.

    Average error vs optimal B&B: ~4.7 % (Table 2 in paper).
    Restricted variant: the original paper uses restricted moves
    (only blockers above a target are moved); this port runs on the
    CRP-D unrestricted environment but only ever moves blockers above
    a selected target container, which is equivalent in practice.
    """

    name                = "Kim–Hong (2006) ENAR [group]"
    category            = "Heuristic"
    description         = (
        "Kim & Hong (2006) §3.2 + §3.3.2 group-priority ENAR heuristic for CRP-D. "
        "Selects the best target container per step by minimising "
        "R[a] = ΔENAR (Eq. 2) + realized relocs over candidate plans. "
        "Avg gap to B&B ≈4.7 %% (paper Table 2). "
        "COR 33 (2006) 940–954."
    )
    compatible_problems = ["CRP-D"]
    geometry            = "single-bay"
    objectives          = ["relocations"]
    fidelity            = "faithful"
    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        cfg     = self.config
        n_seeds = 1  # multi-seed eval removed; single run only
        all_metrics: List[Dict] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.config.seed = getattr(cfg, "seed", 0) + seed

            solution, metrics = _run_episode(env, select_action)
            all_metrics.append(metrics)

            primary = float(metrics.get("relocations", 0.0))
            if primary < self._best_metric:
                self._best_metric   = primary
                self._best_solution = solution[:]

            self._push(
                result_queue,
                step     = seed + 1,
                metric   = primary,
                metrics  = metrics,
                progress = (seed + 1) / n_seeds,
                snapshot = env.get_state_snapshot(),
            )

        if all_metrics:
            agg = {
                k: float(np.mean([m[k] for m in all_metrics if k in m]))
                for k in all_metrics[0]
            }
            self._push(
                result_queue,
                step     = n_seeds,
                metric   = self._best_metric,
                metrics  = agg,
                progress = 1.0,
            )

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution
