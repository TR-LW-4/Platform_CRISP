"""
Poor-move LNS for Wang2026GRASP.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .construction import (
    Move,
    _next_priority_after_target,
    _rr_task1_rc,
    _rr_task2_oc,
    _step_high_action,
    auto_retrieve,
    replay_actions,
    run_greedy_no_deadlock,
)
from .definitions import compute_bad_map, is_rc


# ================================================================ #
#  Poor-move detection                                             #
# ================================================================ #

# Poor-move type codes for the fitness-weighted roulette wheel.
POOR_REL_CMM = 0   # criterion (1.1)
POOR_REL_CMB = 1   # criterion (1.2)
POOR_TGT_LNK = 2   # criterion (2.1)  — link to a poor relocation
POOR_TGT_MC  = 3   # criterion (2.2)  — target retrieval exceeds M_c blockers
NUM_POOR_TYPES = 4


def _target_block_size(records: Sequence[Move]) -> Dict[int, int]:
    """Return {block_idx: number of LOW moves in that block}."""
    sizes: Dict[int, int] = defaultdict(int)
    for r in records:
        if r.kind == "low":
            sizes[r.block_idx] += 1
    return sizes


def find_poor_moves(
    records: Sequence[Move],
    Ma: int,
    Mb: int,
    Mc: int,
) -> List[Tuple[int, int]]:
    """
    Return [(step_index_in_records, POOR_TYPE), ...] for every poor move.
    A move can appear multiple times with different POOR_TYPEs (roulette
    treats them as separate opportunities).
    """
    out: List[Tuple[int, int]] = []
    block_sizes = _target_block_size(records)

    # ── Criterion (1.1) CMM ──
    by_cont: Dict[int, List[int]] = defaultdict(list)
    for i, r in enumerate(records):
        if r.kind == "low" and r.cont_id >= 0:
            by_cont[r.cont_id].append(i)
    for cid, steps in by_cont.items():
        if len(steps) >= Ma:
            out.append((steps[0], POOR_REL_CMM))

    # ── Criterion (1.2) CMB ──
    # For each HIGH (target) block, if it required ≥ Mb blocker moves AND
    # the target was previously moved as a blocker, mark the LAST such
    # previous relocation of that target as poor.
    prev_relocs_of: Dict[int, List[int]] = defaultdict(list)
    for i, r in enumerate(records):
        if r.kind != "high":
            if r.kind == "low" and r.cont_id >= 0:
                prev_relocs_of[r.cont_id].append(i)
            continue
        # HIGH move: look at its LOW block size
        blk_size = block_sizes.get(r.block_idx, 0)
        tid = r.target_id
        if blk_size >= Mb and tid >= 0:
            prior = prev_relocs_of.get(tid, [])
            if prior:
                out.append((prior[-1], POOR_REL_CMB))

    # ── Criterion (2.1) — link poor relocations to their target ──
    high_index_by_block: Dict[int, int] = {}
    for i, r in enumerate(records):
        if r.kind == "high":
            high_index_by_block[r.block_idx] = i
    poor_reloc_steps = {step for step, kind in out
                        if kind in (POOR_REL_CMM, POOR_REL_CMB)}
    for step in poor_reloc_steps:
        blk = records[step].block_idx
        hstep = high_index_by_block.get(blk, -1)
        if hstep >= 0:
            out.append((hstep, POOR_TGT_LNK))

    # ── Criterion (2.2) — target requiring > Mc blocker relocations ──
    for i, r in enumerate(records):
        if r.kind == "high":
            blk_size = block_sizes.get(r.block_idx, 0)
            if blk_size > Mc:
                out.append((i, POOR_TGT_MC))

    # De-duplicate identical (step, kind) entries while preserving order.
    seen: Set[Tuple[int, int]] = set()
    unique: List[Tuple[int, int]] = []
    for pair in out:
        if pair in seen:
            continue
        seen.add(pair)
        unique.append(pair)
    return unique


def _roulette_choice(
    poor: Sequence[Tuple[int, int]],
    fitness: Sequence[float],
    rng: np.random.RandomState,
) -> Tuple[int, int]:
    """Pick one (step, poor_type) proportional to fitness[poor_type]."""
    weights = np.array([max(fitness[kind], 1e-9) for _, kind in poor],
                       dtype=float)
    total = weights.sum()
    if total <= 0:
        return poor[int(rng.randint(0, len(poor)))]
    probs = weights / total
    idx = int(rng.choice(len(poor), p=probs))
    return poor[idx]


# ================================================================ #
#  Neighborhood generation (Algorithm 4)                          #
# ================================================================ #

def _finish_current_target_no_deadlock(
    env,
    current_target,
    rng,
) -> Tuple[List[int], List[Move]]:
    """
    Complete the current HIGH block: keep relocating blockers of the current
    target (using no-deadlock RR) until the target is retrieved.
    """
    extra_actions: List[int] = []
    extra_records: List[Move] = []
    if current_target is None or env._mode != "low":
        return extra_actions, extra_records

    high_idx = env._source_stack_idx  # not the real block idx; recompute if needed
    # We piggy-back block_idx=-1 for these extras; LNS re-analyses records anyway.

    reloc_cnt: Dict[int, int] = defaultdict(int)
    max_iters = env._n_stacks * env.config.max_tiers * 4 + 100
    it = 0
    while env._mode == "low" and it < max_iters:
        it += 1
        src_action = env._source_stack_idx
        src_key    = env._idx_to_stack(src_action)
        src_stk    = env.yard.stacks.get(src_key)
        blocker    = src_stk.top if (src_stk and not src_stk.is_empty) else None
        if blocker is None:
            break

        if is_rc(blocker):
            next_pri = _next_priority_after_target(env, current_target)
            grp = current_target.group
            action = _rr_task1_rc(env, src_action, grp, next_pri,
                                  excluded_actions=None, rng=rng)
        else:
            bad_map = compute_bad_map(env)
            action = _rr_task2_oc(env, blocker, src_action, bad_map,
                                  excluded_actions=None,
                                  use_deadlock=False, rng=rng)

        dst_key = env._idx_to_stack(action)
        reloc_cnt[id(blocker)] += 1
        rec = Move(
            kind="low",
            action=action,
            cont_id=blocker.id,
            src_key=src_key,
            dst_key=dst_key,
            is_bad=False,
            reloc_n=reloc_cnt[id(blocker)],
            target_id=current_target.id,
            block_idx=-1,
        )
        env.step(action)
        extra_actions.append(action)
        extra_records.append(rec)
        auto_retrieve(env)

    return extra_actions, extra_records


def _greedy_complete_partial(
    env,
    rng: np.random.RandomState,
) -> List[int]:
    """
    Complete the current partial episode using Greedy_Randomized_noRRDL.
    Called AFTER the partial prefix has already been applied to env.
    Returns the extra actions.
    """
    from .construction import _select_target   # local import to avoid cycle
    extra: List[int] = []
    max_steps = env._n_stacks * env.config.max_tiers * 6 + 200
    current_target = None
    while not env._done and len(extra) < max_steps:
        mask = env._build_action_mask()
        if not mask.any():
            break

        if env._mode == "high":
            bad_map = compute_bad_map(env)
            action = _select_target(env, bad_map, rng, use_deadlock=False)
            if action < 0:
                break
            current_target = _step_high_action(env, action)
        else:
            src_action = env._source_stack_idx
            src_key    = env._idx_to_stack(src_action)
            src_stk    = env.yard.stacks.get(src_key)
            blocker    = src_stk.top if (src_stk and not src_stk.is_empty) else None
            if blocker is None:
                break
            if is_rc(blocker):
                next_pri = _next_priority_after_target(env, current_target)
                grp = current_target.group if current_target is not None else -1
                action = _rr_task1_rc(env, src_action, grp, next_pri,
                                      excluded_actions=None, rng=rng)
            else:
                bad_map = compute_bad_map(env)
                action = _rr_task2_oc(env, blocker, src_action, bad_map,
                                      excluded_actions=None,
                                      use_deadlock=False, rng=rng)

        _, _, terminated, truncated, _ = env.step(action)
        extra.append(action)
        auto_retrieve(env)
        if terminated or truncated:
            break
    return extra


def neigh_gene(
    env,
    actions: List[int],
    records: List[Move],
    fitness: Sequence[float],
    rng: np.random.RandomState,
    Ma: int,
    Mb: int,
    Mc: int,
) -> Tuple[Optional[List[int]], Optional[int]]:
    """
    Generate one neighbor from (actions, records).
    Returns (new_actions, poor_type).  new_actions is None if no poor move
    exists or no valid neighbor could be produced.
    """
    poor = find_poor_moves(records, Ma, Mb, Mc)
    if not poor:
        return None, None

    selected_step, poor_type = _roulette_choice(poor, fitness, rng)
    sel_move = records[selected_step]

    # Determine the prefix to keep.
    if sel_move.kind == "low":
        # Poor relocation: drop moves ≥ selected_step
        prefix = actions[:selected_step]
        # If the immediately preceding move is also LOW → the parent target
        # retrieval was interrupted mid-way.  We must finish it (using
        # no-deadlock RR) BEFORE re-completing the rest of the plan.
        need_finish = (
            selected_step - 1 >= 0
            and records[selected_step - 1].kind == "low"
        )
    else:
        # Poor target: drop back to (and including) the LAST HIGH move
        # immediately preceding the poor target.  This wipes out both the
        # poor HIGH decision and the LOW moves it triggered.
        prev_high = -1
        for k in range(selected_step - 1, -1, -1):
            if records[k].kind == "high":
                prev_high = k
                break
        # Everything BEFORE prev_high is fully applied; the poor target
        # itself (and any subsequent moves) is discarded.
        prefix = actions[:prev_high + 1] if prev_high >= 0 else []
        need_finish = False

    # Apply prefix to env.
    env.reset()
    auto_retrieve(env)
    for a in prefix:
        if env._done:
            break
        env.step(a)
        auto_retrieve(env)
    if env._done:
        return prefix, poor_type

    extra_actions: List[int] = []
    if need_finish and env._mode == "low":
        # Find the current target that env is trying to reach.
        src_action = env._source_stack_idx
        src_key    = env._idx_to_stack(src_action)
        src_stk    = env.yard.stacks.get(src_key)
        current_target = None
        if src_stk is not None and not src_stk.is_empty:
            for c in reversed(src_stk.containers):
                if env._is_retrievable(c):
                    current_target = c
                    break
        finish_actions, _ = _finish_current_target_no_deadlock(
            env, current_target, rng
        )
        extra_actions.extend(finish_actions)

    # Now complete the rest of the plan.
    tail = _greedy_complete_partial(env, rng)
    new_actions = prefix + extra_actions + tail
    return new_actions, poor_type


# ================================================================ #
#  LNS (Algorithm 3)                                               #
# ================================================================ #

@dataclass
class LNSParams:
    Ma:           int   = 2
    Mb:           int   = 2
    Mc:           int   = 3
    noimprlimit:  int   = 3
    init_fitness: float = 1.0
    fit_alpha:    float = 1.5
    init_temp:    float = 10.0
    final_temp:   float = 0.1
    cooling:      float = 0.95
    boltzmann:    float = 1.0


def _relocs_of(actions: List[int], env_factory) -> Tuple[int, List[Move]]:
    env = env_factory()
    records, metrics = replay_actions(env, actions)
    return int(metrics.get("relocations", 0)), records


def run_lns(
    env_factory,
    init_actions: List[int],
    init_records: List[Move],
    rng:          np.random.RandomState,
    params:       LNSParams,
) -> Tuple[List[int], List[Move], int]:
    """
    Local search around ``init_actions``.

    Returns (best_actions, best_records, best_relocations).
    """
    curr_actions = list(init_actions)
    curr_records = list(init_records)
    curr_relocs  = sum(1 for r in curr_records if r.kind == "low")

    best_actions = curr_actions
    best_records = curr_records
    best_relocs  = curr_relocs

    fitness = [params.init_fitness] * NUM_POOR_TYPES
    noimpr = 0
    temp   = params.init_temp

    while noimpr < params.noimprlimit:
        env = env_factory()
        env.reset()   # ensure clean state
        auto_retrieve(env)
        new_actions, poor_type = neigh_gene(
            env, curr_actions, curr_records, fitness, rng,
            params.Ma, params.Mb, params.Mc,
        )
        if new_actions is None:
            break

        new_relocs, new_records = _relocs_of(new_actions, env_factory)

        if new_relocs < best_relocs:
            best_actions = new_actions
            best_records = new_records
            best_relocs  = new_relocs
            curr_actions = new_actions
            curr_records = new_records
            curr_relocs  = new_relocs
            if poor_type is not None:
                fitness[poor_type] = fitness[poor_type] * params.fit_alpha
            noimpr = 0
        else:
            delta = new_relocs - curr_relocs
            accept = False
            if delta <= 0:
                accept = True
            elif temp > params.final_temp:
                prob = math.exp(-delta / max(params.boltzmann * temp, 1e-9))
                accept = (rng.random() < prob)
            if accept:
                curr_actions = new_actions
                curr_records = new_records
                curr_relocs  = new_relocs
            noimpr += 1
            temp *= params.cooling

    return best_actions, best_records, best_relocs
