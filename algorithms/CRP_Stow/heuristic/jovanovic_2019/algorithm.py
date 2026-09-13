"""
JovanovicGRASP
<2019> <heuristic> <stowage> <multi-bay> <CRP-Stow>
Greedy-with-correction and GRASP for the BRLP

------------------------------- Reference --------------------------------
R. Jovanović, S. Tanaka, T. Nishi, S. Voß,
"A GRASP approach for solving the Blocks Relocation Problem with Stowage Plan",
Flexible Services and Manufacturing Journal 31 (2019) 702–729.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import multiprocessing as mp
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig


# ================================================================ #
#  Well-located & due-date infrastructure                          #
# ================================================================ #

def _is_well_located(c, containers_below) -> bool:
    """c is well-located iff no d below it (same stack) has vs(d)==vs(c), vt(d)<vt(c)."""
    vs_c, vt_c = c.group, c.priority
    return not any(d.group == vs_c and d.priority < vt_c for d in containers_below)


def _compute_well_located(env) -> Dict[int, bool]:
    """Return {id(c): bool} well-located status for all containers currently in yard."""
    wl: Dict[int, bool] = {}
    for stk in env.yard.stacks.values():
        cs = stk.containers
        for i, c in enumerate(cs):
            wl[id(c)] = _is_well_located(c, cs[:i])
    return wl


def _dd_stk(stk, vs_c: int, N: int) -> int:
    """
    dd(S, c): min vessel tier among containers in S sharing vessel stack vs_c.
    Returns N+1 if stack is empty or no container has vs == vs_c.
    """
    if stk is None or stk.is_empty:
        return N + 1
    same = [ct.priority for ct in stk.containers if ct.group == vs_c]
    return min(same) if same else N + 1


def _cdd_stk(stk, vessel_loaded: List[int]) -> int:
    """cdd(S): min current-due-date of containers in S. cdd(c) = vt(c) − loaded[vs(c)]."""
    if stk is None or stk.is_empty:
        return 0
    return min(ct.priority - vessel_loaded[ct.group] for ct in stk.containers)


# ================================================================ #
#  4-cycle detection (§4.2–4.3)                                   #
# ================================================================ #

def _creates_relevant_4_cycle(
    env,
    c_group: int,
    c_priority: int,
    dst_key: Tuple[int, int],
    wl: Dict[int, bool],
) -> bool:
    """
    True iff placing (c_group, c_priority) on top of dst_key would create
    a relevant 4-cycle.

    The only possible pattern for a newly placed container c (which is on top
    of dst_key and therefore has no incoming yard edges) is:

        c →_y d →_v d' →_y x →_v c

    where all four nodes (c, d, d', x) are well-located and edges alternate
    between yard (→_y) and vessel (→_v) precedence.
    """
    dst = env.yard.stacks.get(dst_key)
    if dst is None or dst.is_empty:
        return False

    # c must be well-located after placement in dst_key
    if any(d.group == c_group and d.priority < c_priority for d in dst.containers):
        return False

    # Pattern: c →_y d (d is below c in dst_key)
    #          d →_v d' (same vessel stack as d, higher vt)
    #          d' →_y x (x is below d' in some yard stack)
    #          x →_v c  (same vessel stack as c, vt(x) < c_priority)
    for d in dst.containers:
        if not wl.get(id(d), False):
            continue
        vs_d, vt_d = d.group, d.priority
        for stk2 in env.yard.stacks.values():
            cs2 = stk2.containers
            for j, dp in enumerate(cs2):
                if dp.group != vs_d or dp.priority <= vt_d:
                    continue
                if not wl.get(id(dp), False):
                    continue
                # d' is at position j; containers below d' are cs2[:j]
                for x in cs2[:j]:
                    if (x.group == c_group
                            and x.priority < c_priority
                            and wl.get(id(x), False)):
                        return True
    return False


def _is_4_blocking(
    env,
    c,
    c_pos: int,
    stk_key: Tuple[int, int],
    wl: Dict[int, bool],
) -> bool:
    """
    True iff well-located container c (at position c_pos from bottom in stk_key)
    participates in a relevant 4-cycle as a blocking node (has outgoing yard edge).

    Same pattern as above with c playing the role of the blocking node.
    """
    if not wl.get(id(c), False):
        return False
    vs_c, vt_c = c.group, c.priority
    stk = env.yard.stacks.get(stk_key)
    if stk is None:
        return False

    for d in stk.containers[:c_pos]:  # containers below c
        if not wl.get(id(d), False):
            continue
        vs_d, vt_d = d.group, d.priority
        for stk2 in env.yard.stacks.values():
            cs2 = stk2.containers
            for j, dp in enumerate(cs2):
                if dp.group != vs_d or dp.priority <= vt_d:
                    continue
                if not wl.get(id(dp), False):
                    continue
                for x in cs2[:j]:
                    if (x.group == vs_c
                            and x.priority < vt_c
                            and wl.get(id(x), False)):
                        return True
    return False


# ================================================================ #
#  HR — Relocation heuristics (§5.1)                              #
# ================================================================ #

def _iter_dsts(env, src_key):
    """Yield (stk, key, bay, row, idx) for every valid destination stack."""
    n = env._n_stacks
    for i in range(n):
        key = env._idx_to_stack(i)
        if key == src_key:
            continue
        stk = env.yard.stacks.get(key)
        if stk is not None and not stk.is_full:
            yield stk, key, key[0], key[1], i


def _pick(pool: list, rng, key_len: int = 2) -> int:
    """
    Sort pool, randomly select among entries tied on the first key_len fields.
    Returns the last field of the chosen entry (the action index).
    Returns 0 if pool is empty.
    """
    if not pool:
        return 0
    pool.sort()
    if rng is None:
        return pool[0][-1]
    min_key = tuple(pool[0][:key_len])
    tied = [p[-1] for p in pool if tuple(p[:key_len]) == min_key]
    return int(rng.choice(tied))


def _select_low_MinMax4CB(
    env,
    src_key: Tuple[int, int],
    c,
    wl: Dict[int, bool],
    N: int,
    rng=None,
    except_keys: Optional[Set] = None,
) -> int:
    """
    MinMax4CB relocation heuristic (HR, §5.1).

    Candidates are split into:
      wl_pool  – stacks where c becomes well-located (dd(S,c) > vt(c))
      nwl_pool – all others

    Within wl_pool: key = (dd(S,c) + M·Creates4Cycle(S), −cdd(S))
    Within nwl_pool: key = (−dd(S,c), −cdd(S))  →  maximise dd, then cdd

    When except_keys excludes all candidates, fall back ignoring except_keys.
    Returns -1 if truly no destination exists.
    """
    if c is None:
        return 0
    vs_c, vt_c = c.group, c.priority
    M = N + 1
    except_keys = except_keys or set()
    wl_pool: list = []
    nwl_pool: list = []

    for stk, key, bay, row, i in _iter_dsts(env, src_key):
        if key in except_keys:
            continue
        dd  = _dd_stk(stk, vs_c, N)
        cdd = _cdd_stk(stk, env._vessel_loaded)
        if dd > vt_c:
            c4  = _creates_relevant_4_cycle(env, vs_c, vt_c, key, wl)
            wl_pool.append((dd + M * int(c4), -cdd, bay, row, i))
        else:
            nwl_pool.append((-dd, -cdd, bay, row, i))

    pool = wl_pool if wl_pool else nwl_pool
    if not pool:
        # All candidates excluded: ignore except_keys, try again
        for stk, key, bay, row, i in _iter_dsts(env, src_key):
            dd  = _dd_stk(stk, vs_c, N)
            cdd = _cdd_stk(stk, env._vessel_loaded)
            if dd > vt_c:
                wl_pool.append((dd, -cdd, bay, row, i))
            else:
                nwl_pool.append((-dd, -cdd, bay, row, i))
        pool = wl_pool if wl_pool else nwl_pool
        if not pool:
            return -1

    return _pick(pool, rng)


def _select_low_MinMax(
    env,
    src_key: Tuple[int, int],
    c,
    wl: Dict[int, bool],
    N: int,
    rng=None,
    except_keys: Optional[Set] = None,
) -> int:
    """MinMax relocation heuristic (HR, §5.1) without 4-cycle detection."""
    if c is None:
        return 0
    vs_c, vt_c = c.group, c.priority
    except_keys = except_keys or set()
    wl_pool: list = []
    nwl_pool: list = []

    for stk, key, bay, row, i in _iter_dsts(env, src_key):
        if key in except_keys:
            continue
        dd  = _dd_stk(stk, vs_c, N)
        cdd = _cdd_stk(stk, env._vessel_loaded)
        if dd > vt_c:
            wl_pool.append((dd, -cdd, bay, row, i))
        else:
            nwl_pool.append((-dd, -cdd, bay, row, i))

    pool = wl_pool if wl_pool else nwl_pool
    if not pool:
        return 0
    return _pick(pool, rng)


def _select_low_LT(
    env,
    src_key: Tuple[int, int],
    c,
    wl: Dict[int, bool],
    N: int,
    rng=None,
    except_keys: Optional[Set] = None,
) -> int:
    """Lowest Tier: relocate to stack with fewest containers."""
    except_keys = except_keys or set()
    pool = []
    for stk, key, bay, row, i in _iter_dsts(env, src_key):
        if key in except_keys:
            continue
        pool.append((stk.height, bay, row, i))
    if not pool:
        return 0
    return _pick(pool, rng)


# ================================================================ #
#  HL — Retrieval heuristics (§5.2)                               #
# ================================================================ #

def _topmost_retrievable_pos(env, stk) -> int:
    """Return 0-based index (from bottom) of the topmost retrievable container, or -1."""
    pos = -1
    for i, c in enumerate(stk.containers):
        if env._is_retrievable(c):
            pos = i   # keep updating → final value is topmost
    return pos


def _select_high_MinW4CB(
    env,
    wl: Dict[int, bool],
    rng=None,
    M: int = 10_000,
    K: int = 100,
) -> int:
    """
    MinW4CB retrieval heuristic (HL, §5.2).
    Score = M·AboveWL* + K·Above4CBlocking + AboveNWL  (minimise).
    """
    pool: list = []
    n = env._n_stacks
    for i in range(n):
        key = env._idx_to_stack(i)
        stk = env.yard.stacks.get(key)
        if stk is None or stk.is_empty:
            continue
        tp = _topmost_retrievable_pos(env, stk)
        if tp < 0:
            continue
        cs = stk.containers
        above_wl = above_4cb = above_nwl = 0
        for pos, ca in enumerate(cs[tp + 1:], start=tp + 1):
            if wl.get(id(ca), False):
                if _is_4_blocking(env, ca, pos, key, wl):
                    above_4cb += 1
                else:
                    above_wl += 1
            else:
                above_nwl += 1
        score = M * above_wl + K * above_4cb + above_nwl
        pool.append((score, key[0], key[1], i))

    return _pick(pool, rng, key_len=1)


def _select_high_MinW(
    env,
    wl: Dict[int, bool],
    rng=None,
    M: int = 10_000,
) -> int:
    """MinW retrieval heuristic (HL, §5.2) without 4-cycle information."""
    pool: list = []
    n = env._n_stacks
    for i in range(n):
        key = env._idx_to_stack(i)
        stk = env.yard.stacks.get(key)
        if stk is None or stk.is_empty:
            continue
        tp = _topmost_retrievable_pos(env, stk)
        if tp < 0:
            continue
        cs = stk.containers
        above_wl  = sum(1 for ca in cs[tp + 1:] if wl.get(id(ca), False))
        above_nwl = sum(1 for ca in cs[tp + 1:] if not wl.get(id(ca), False))
        pool.append((M * above_wl + above_nwl, key[0], key[1], i))

    return _pick(pool, rng, key_len=1)


def _select_high_MinB(env, wl=None, rng=None) -> int:
    """MinB: minimum blockers above topmost retrievable (simplest HL)."""
    pool: list = []
    n = env._n_stacks
    for i in range(n):
        key = env._idx_to_stack(i)
        stk = env.yard.stacks.get(key)
        if stk is None or stk.is_empty:
            continue
        tp = _topmost_retrievable_pos(env, stk)
        if tp < 0:
            continue
        blockers = len(stk.containers) - 1 - tp
        pool.append((blockers, key[0], key[1], i))

    return _pick(pool, rng, key_len=1)


# ================================================================ #
#  Move recording                                                  #
# ================================================================ #

@dataclass
class _MoveRec:
    step: int
    mode: str           # 'high' or 'low'
    action: int
    cont_id: int   = -1
    dst_key: tuple = ()
    well_located: bool = False
    reloc_n: int   = 0   # cumulative relocations of this container up to & incl. this move
    block_idx: int = -1  # index of the parent HIGH step


# ================================================================ #
#  Episode runners                                                 #
# ================================================================ #

def _run_recorded(
    env,
    hl_fn: Callable,
    hr_fn: Callable,
    rng=None,
) -> Tuple[List[int], List[_MoveRec], Dict]:
    """
    Run one greedy episode, recording move metadata.
    Returns (actions, records, metrics).
    """
    env.reset()
    actions: List[int] = []
    records: List[_MoveRec] = []
    done = False
    N = len(env.containers)
    max_steps = env._n_stacks * getattr(env.config, "max_tiers", 10) * 6 + 200
    wl = _compute_well_located(env)
    reloc_cnt: Dict[int, int] = defaultdict(int)
    high_idx = -1

    while not done and len(actions) < max_steps:
        mask = env._build_action_mask()
        if not mask.any():
            break
        step = len(actions)

        if env._mode == "high":
            high_idx += 1
            wl = _compute_well_located(env)
            action = hl_fn(env, wl, rng)
            rec = _MoveRec(step=step, mode="high", action=action,
                           block_idx=high_idx)
        else:
            src_key = env._idx_to_stack(env._source_stack_idx)
            src_stk = env.yard.stacks.get(src_key)
            c       = src_stk.top if (src_stk and not src_stk.is_empty) else None
            action  = hr_fn(env, src_key, c, wl, N, rng)
            if action < 0:
                action = 0
            dst_key = env._idx_to_stack(action)

            c_wl = False
            if c is not None:
                dst_stk  = env.yard.stacks.get(dst_key)
                dst_below = list(dst_stk.containers) if (dst_stk and not dst_stk.is_empty) else []
                c_wl = _is_well_located(c, dst_below)
                reloc_cnt[id(c)] += 1

            rec = _MoveRec(
                step=step, mode="low", action=action,
                cont_id=id(c) if c else -1,
                dst_key=dst_key,
                well_located=c_wl,
                reloc_n=reloc_cnt[id(c)] if c else 0,
                block_idx=high_idx,
            )

        _, _, terminated, truncated, _ = env.step(action)
        actions.append(action)
        records.append(rec)
        done = terminated or truncated

    return actions, records, env.get_metrics()


def _replay_and_record(env, actions: List[int]) -> Tuple[List[_MoveRec], Dict]:
    """Replay a stored action sequence and record move metadata."""
    env.reset()
    records: List[_MoveRec] = []
    reloc_cnt: Dict[int, int] = defaultdict(int)
    high_idx = -1

    for step, action in enumerate(actions):
        if env._done:
            break
        if env._mode == "high":
            high_idx += 1
            rec = _MoveRec(step=step, mode="high", action=action,
                           block_idx=high_idx)
        else:
            src_key = env._idx_to_stack(env._source_stack_idx)
            src_stk = env.yard.stacks.get(src_key)
            c       = src_stk.top if (src_stk and not src_stk.is_empty) else None
            dst_key = env._idx_to_stack(action)
            c_wl = False
            if c is not None:
                dst_stk  = env.yard.stacks.get(dst_key)
                dst_below = list(dst_stk.containers) if (dst_stk and not dst_stk.is_empty) else []
                c_wl = _is_well_located(c, dst_below)
                reloc_cnt[id(c)] += 1
            rec = _MoveRec(
                step=step, mode="low", action=action,
                cont_id=id(c) if c else -1,
                dst_key=dst_key,
                well_located=c_wl,
                reloc_n=reloc_cnt[id(c)] if c else 0,
                block_idx=high_idx,
            )
        records.append(rec)
        env.step(action)

    return records, env.get_metrics()


def _greedy_complete(env, hl_fn: Callable, hr_fn: Callable, rng=None) -> List[int]:
    """Run greedy from the current env state until done. Returns additional actions."""
    addl: List[int] = []
    done = env._done
    N    = len(env.containers)
    max_steps = env._n_stacks * getattr(env.config, "max_tiers", 10) * 6 + 200
    wl   = _compute_well_located(env)

    while not done and len(addl) < max_steps:
        mask = env._build_action_mask()
        if not mask.any():
            break
        if env._mode == "high":
            wl = _compute_well_located(env)
            action = hl_fn(env, wl, rng)
        else:
            src_key = env._idx_to_stack(env._source_stack_idx)
            src_stk = env.yard.stacks.get(src_key)
            c       = src_stk.top if (src_stk and not src_stk.is_empty) else None
            action  = hr_fn(env, src_key, c, wl, N, rng)
            if action < 0:
                action = 0
        _, _, terminated, truncated, _ = env.step(action)
        addl.append(action)
        done = terminated or truncated

    return addl


# ================================================================ #
#  Suspicious-move identification (§5.3.1)                        #
# ================================================================ #

def _suspicious_steps(records: List[_MoveRec], MM: int = 2, MB: int = 1) -> Set[int]:
    """
    Return set of step indices that are suspicious LOW moves.

    CMM – first relocation of a container relocated ≥ MM times.
    CMB – last relocation before retrieval of a previously-relocated container,
          when that retrieval required ≥ MB other relocations.
    CMS – a well-located relocation that is undone before the container is retrieved.
    """
    susp: Set[int] = set()
    low_recs = [r for r in records if r.mode == "low" and r.cont_id >= 0]

    # --- CMM ---
    by_cont: Dict[int, List[int]] = defaultdict(list)
    for r in low_recs:
        by_cont[r.cont_id].append(r.step)
    for steps in by_cont.values():
        if len(steps) >= MM:
            susp.add(steps[0])

    # --- CMB ---
    # Group LOW moves by HIGH-step block
    by_block: Dict[int, List[_MoveRec]] = defaultdict(list)
    for r in low_recs:
        by_block[r.block_idx].append(r)
    for lows in by_block.values():
        if len(lows) < MB:
            continue
        for r in lows:
            if r.reloc_n > 1:   # container was relocated before this block
                same = [x for x in lows if x.cont_id == r.cont_id]
                if same:
                    susp.add(max(same, key=lambda x: x.step).step)

    # --- CMS ---
    wl_at: Dict[int, int] = {}
    for r in low_recs:
        if r.well_located:
            wl_at[r.cont_id] = r.step
        elif r.cont_id in wl_at:
            susp.add(wl_at.pop(r.cont_id))

    return susp


# ================================================================ #
#  Correction procedure (Algorithm 2)                             #
# ================================================================ #

def _correct(
    env,
    actions: List[int],
    records: List[_MoveRec],
    hl_fn: Callable,
    hr_fn: Callable,
    MM: int = 2,
    MB: int = 1,
) -> Tuple[List[int], Dict]:
    """
    Correction procedure (Jovanović 2019, Algorithm 2).
    Iterates through suspicious LOW moves, tries alternative destinations,
    and re-completes greedily.  Restarts from the beginning whenever an
    improvement is found.
    Returns (best_actions, metrics).
    """
    best = list(actions)
    best_relocs = int(sum(1 for r in records if r.mode == "low"))
    susp = _suspicious_steps(records, MM, MB)
    if not susp:
        return best, env.evaluate(best)

    i = 0
    while i < len(best):
        if i >= len(records):
            break
        rec = records[i]
        if rec.mode != "low" or rec.step not in susp:
            i += 1
            continue

        except_keys: Set = {rec.dst_key}
        improved = False

        while True:
            # Replay partial solution up to (not including) step i
            env.reset()
            valid = True
            for a in best[:i]:
                if env._done:
                    valid = False
                    break
                env.step(a)
            if not valid or env._mode != "low":
                break

            src_key = env._idx_to_stack(env._source_stack_idx)
            src_stk = env.yard.stacks.get(src_key)
            c = src_stk.top if (src_stk and not src_stk.is_empty) else None
            if c is None:
                break

            N   = len(env.containers)
            wl  = _compute_well_located(env)
            new = hr_fn(env, src_key, c, wl, N, None, except_keys)
            if new < 0:
                break   # no more alternatives outside except_keys

            new_key  = env._idx_to_stack(new)
            new_stk  = env.yard.stacks.get(new_key)
            new_below = list(new_stk.containers) if (new_stk and not new_stk.is_empty) else []
            new_wl   = _is_well_located(c, new_below)

            env.step(new)
            completion = _greedy_complete(env, hl_fn, hr_fn, rng=None)
            candidate  = best[:i] + [new] + completion
            cand_relocs = int(env.get_metrics()["relocations"])

            if cand_relocs < best_relocs:
                best = candidate
                best_relocs = cand_relocs
                records, _ = _replay_and_record(env, best)
                susp = _suspicious_steps(records, MM, MB)
                i = 0
                improved = True
                break

            except_keys.add(new_key)
            if not new_wl:
                break   # Algorithm 2: stop trying when alternative is not well-located

        if not improved:
            i += 1

    return best, env.evaluate(best)


# ================================================================ #
#  High-level episode entry points                                #
# ================================================================ #

def _make_hl(use_4cb: bool) -> Callable:
    if use_4cb:
        return lambda env, wl, rng=None: _select_high_MinW4CB(env, wl, rng)
    return lambda env, wl, rng=None: _select_high_MinW(env, wl, rng)


def _make_hr(use_4cb: bool) -> Callable:
    if use_4cb:
        return lambda env, src_key, c, wl, N, rng=None, ek=None: \
            _select_low_MinMax4CB(env, src_key, c, wl, N, rng, ek)
    return lambda env, src_key, c, wl, N, rng=None, ek=None: \
        _select_low_MinMax(env, src_key, c, wl, N, rng, ek)


def _run_grc(env, use_4cb: bool = True, MM: int = 2, MB: int = 1):
    """GR-C: deterministic greedy + correction.  Returns (actions, metrics)."""
    hl = _make_hl(use_4cb)
    hr = _make_hr(use_4cb)
    actions, records, _ = _run_recorded(env, hl, hr, rng=None)
    return _correct(env, actions, records, hl, hr, MM, MB)


def _run_grasp(
    env,
    n_iter: int = 100,
    use_4cb: bool = True,
    MM: int = 2,
    MB: int = 1,
    seed: Optional[int] = None,
    progress_cb: Optional[Callable] = None,
) -> Tuple[List[int], Dict]:
    """GRASP: n_iter randomised GR-C runs.  Returns (best_actions, best_metrics)."""
    rng = np.random.RandomState(seed)
    hl  = _make_hl(use_4cb)
    hr  = _make_hr(use_4cb)
    best_actions: Optional[List[int]] = None
    best_metrics: Optional[Dict] = None
    best_relocs  = float("inf")

    for it in range(n_iter):
        acts, recs, _ = _run_recorded(env, hl, hr, rng)
        cand_acts, cand_metrics = _correct(env, acts, recs, hl, hr, MM, MB)
        r = float(cand_metrics.get("relocations", float("inf")))
        if r < best_relocs:
            best_relocs  = r
            best_actions = cand_acts
            best_metrics = cand_metrics
        if progress_cb:
            progress_cb(it + 1, best_relocs)

    return (best_actions or []), (best_metrics or {})


# ================================================================ #
#  Platform algorithm classes                                      #
# ================================================================ #

class _JovanovicBase(BaseAlgorithm):
    """Shared train() scaffold for Jovanović (2019) algorithms."""

    compatible_problems = ["CRP-Stow"]
    geometry = "multi-bay"
    objectives = ["relocations"]
    fidelity = "faithful"
    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    def _run_one(self, env, seed: Optional[int] = None) -> Tuple[List[int], Dict]:
        raise NotImplementedError

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
            env.config.seed = cfg.seed + seed

            sol, metrics = self._run_one(env, seed=cfg.seed + seed)
            all_metrics.append(metrics)

            primary = float(metrics.get("relocations", 0.0))
            if primary < self._best_metric:
                self._best_metric   = primary
                self._best_solution = sol[:]

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


class JovanovicGRC(_JovanovicBase):
    """
    Jovanović et al. (2019) Greedy with Correction (GR-C) for BRLP.

    Heuristic combination: MinMax4CB (HR) + MinW4CB (HL).
    """

    name        = "Jovanović (2019) GR-C"
    category    = "Heuristic"
    description = (
        "Jovanović et al. (2019) Greedy with Correction for BRLP. "
        "Runs a greedy pass with MinMax4CB (relocation) and MinW4CB (retrieval), "
        "then applies a correction procedure (CMM/CMB/CMS criteria) to reduce "
        "rehandles by trying alternatives at suspicious moves."
    )

    def _run_one(self, env, seed=None):
        return _run_grc(env, use_4cb=True)


class JovanovicGRASP(_JovanovicBase):
    """
    Jovanović et al. (2019) full GRASP for BRLP.

    Runs n_grasp_iters randomised GR-C passes per instance; returns the best
    solution found.  Paper uses 100 iterations; 10–20 gives most of the gain.
    """

    name        = "Jovanović (2019) GRASP"
    category    = "Heuristic"
    description = (
        "Jovanović et al. (2019) GRASP for BRLP.  "
        "Repeats randomised GR-C (MinMax4CB + MinW4CB) for n_grasp_iters iterations "
        "and returns the best solution found.  "
        "Typically 10–15% better than GR-C on large instances; "
        "paper default is 100 iterations."
    )

    def _run_one(self, env, seed=None):
        n_iter = int(getattr(self.config, "n_grasp_iters", 20))
        return _run_grasp(env, n_iter=n_iter, use_4cb=True, seed=seed)

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "n_grasp_iters": {
                "type": "int", "default": 20, "min": 1, "max": 500,
                "label": "GRASP iterations",
                "help": "Randomised GR-C passes per instance (paper uses 100).",
            },
        })
        return base
