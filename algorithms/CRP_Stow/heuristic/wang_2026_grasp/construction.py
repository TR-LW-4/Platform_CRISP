"""
TR/RR constructive greedy for Wang2026GRASP.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from .definitions import (
    _HUGE,
    candidate_stacks,
    classify_above_target,
    compute_bad_map,
    creates_deadlock_if_placed,
    empty_or_rolled_stacks,
    is_bad_if_placed_on,
    is_rc,
    se_of,
    sec_of,
    src_stacks,
    stack_has_oc,
    stack_only_rcs,
    topmost_retrievable_pos,
)


# ================================================================ #
#  Move record                                                     #
# ================================================================ #

@dataclass
class Move:
    """One primitive move (a `high` retrieval decision or a `low` relocation)."""
    kind:      str           # "high" or "low"
    action:    int           # env action index (stack idx)
    cont_id:   int   = -1    # for "low": id of the relocated blocker
    src_key:   Tuple[int, int] = (-1, -1)
    dst_key:   Tuple[int, int] = (-1, -1)
    is_bad:    bool = False  # blocker becomes a bad container after placement
    reloc_n:   int  = 0      # cumulative # of relocations of this container up to & incl. this move
    target_id: int  = -1     # id of the current TARGET container being cleared
    block_idx: int  = -1     # index of the parent HIGH block


# ================================================================ #
#  Auto-Retrieve                                                   #
# ================================================================ #

def auto_retrieve(env) -> None:
    """
    Repeatedly retrieve any candidate that is currently topmost.
    Uses env._do_retrieve directly so no env.step() is consumed.
    (env's own reset also calls _advance_auto_retrievals, so this is normally
    a no-op right after reset; we still call it after every relocation to
    account for cascaded free retrievals.)
    """
    env._advance_auto_retrievals()


# ================================================================ #
#  Function TR (§5.2.2) — target selection                        #
# ================================================================ #

def _select_target(
    env,
    bad_map:      Dict[int, bool],
    rng           = None,
    use_deadlock: bool = True,
) -> int:
    """
    Return the env action index (stack idx) whose topmost retrievable OC
    should be targeted next.  Follows §5.2.2 lexicographic order:

        min NBC(c)   →   max DC(c)   →   min BC(c) + RCB(c)
        →   prefer targets that free up an empty / rolled stack
        →   arbitrary (deterministic: smallest key; randomized: rng.choice)

    ``use_deadlock=False`` disables the DC tie-breaker AND the deadlock check
    within RR when propagated downstream — matches the "setting B" of §6.3.1
    plus the noRRDL simplification of §5.3.3.
    """
    cands = candidate_stacks(env)   # [(action_idx, key, tp)]
    if not cands:
        return -1

    scored: List[Tuple[Tuple[int, int, int], Tuple[int, Tuple[int, int]]]] = []
    for action, key, tp in cands:
        stk = env.yard.stacks[key]
        nbc, dc, bc, rcb = classify_above_target(env, stk, tp, bad_map)
        if not use_deadlock:
            # Merge deadlock nodes back into NBC when deadlock heuristic is off.
            nbc += dc
            dc   = 0
        # Lex key: (min NBC, -DC, min BC+RCB) — python sorts ascending, so
        # negate DC to maximise it.
        lex = (nbc, -dc, bc + rcb)
        scored.append((lex, (action, key)))

    scored.sort(key=lambda t: t[0])
    best_lex = scored[0][0]
    tied = [item for item in scored if item[0] == best_lex]

    if len(tied) == 1:
        return tied[0][1][0]

    # ── §5.2.2 last tie-breaker: prefer creating empty / rolled sinks ──
    # Skip this tie-breaker if the yard already has at least one such sink.
    have_sink = len(empty_or_rolled_stacks(env)) > 0
    if not have_sink:
        can_create: List[Tuple[int, Tuple[int, int]]] = []
        for _, (action, key) in tied:
            stk = env.yard.stacks[key]
            # After retrieving the topmost retrievable OC, would this stack
            # become empty or rolled (only-RC)?
            tp = topmost_retrievable_pos(env, stk)
            remaining = stk.containers[:tp] + stk.containers[tp + 1:]
            if not remaining or all(is_rc(c) for c in remaining):
                can_create.append((action, key))
        pool = can_create if can_create else [item[1] for item in tied]
    else:
        pool = [item[1] for item in tied]

    if rng is None:
        pool.sort(key=lambda x: x[1])
        return pool[0][0]
    return int(pool[int(rng.randint(0, len(pool)))][0])


# ================================================================ #
#  Function RR (§5.2.3) — relocation destination selection        #
# ================================================================ #

def _rr_task1_rc(
    env,
    src_action: int,
    target_group: int,
    target_group_next_priority: Optional[int],
    excluded_actions: Optional[Set[int]] = None,
    rng = None,
) -> int:
    """
    Task 1 (§5.2.3): a Rolled Container is being relocated.
    Preference:
        R1: any empty or non-full rolled stack
        R2: any non-full stack NOT containing a *soon-to-be-retrieved* OC
        R3: any non-full stack
    """
    excluded_actions = excluded_actions or set()
    src_key = env._idx_to_stack(src_action)

    def _valid_dst_stacks() -> List[Tuple[int, Tuple[int, int]]]:
        n = env._n_stacks
        out: List[Tuple[int, Tuple[int, int]]] = []
        for i in range(n):
            if i == src_action or i in excluded_actions:
                continue
            key = env._idx_to_stack(i)
            stk = env.yard.stacks.get(key)
            if stk is None or stk.is_full:
                continue
            out.append((i, key))
        return out

    dsts = _valid_dst_stacks()
    if not dsts:
        # Absolute fallback: ignore excluded_actions.
        for i in range(env._n_stacks):
            if i == src_action:
                continue
            key = env._idx_to_stack(i)
            stk = env.yard.stacks.get(key)
            if stk is not None and not stk.is_full:
                return i
        return src_action   # only src has space – illegal but non-crashing

    # R1
    r1 = [(i, k) for i, k in dsts
          if env.yard.stacks[k].is_empty or stack_only_rcs(env.yard.stacks[k])]
    if r1:
        return _pick(r1, rng)

    # R2 — avoid stacks holding a candidate or the "next candidate after target"
    def _has_soon_oc(stk) -> bool:
        for c in stk.containers:
            if is_rc(c):
                continue
            if env._is_retrievable(c):
                return True
            if (target_group_next_priority is not None
                    and c.group == target_group
                    and c.priority == target_group_next_priority):
                return True
        return False

    r2 = [(i, k) for i, k in dsts if not _has_soon_oc(env.yard.stacks[k])]
    if r2:
        return _pick(r2, rng)

    # R3 — any non-full non-source stack
    return _pick(dsts, rng)


def _rr_task2_oc(
    env,
    blocker,
    src_action: int,
    bad_map:    Dict[int, bool],
    excluded_actions: Optional[Set[int]] = None,
    use_deadlock: bool = True,
    rng = None,
) -> int:
    """
    Task 2 (§5.2.3): an Ordinary Container is being relocated.
    Preference:
        R1: argmax_s se(s)  s ∈ SRC, and blocker→s creates neither a bad
            container nor a deadlock
        R2: argmax_s se(s)  s ∈ SRC, and blocker→s is not a bad container
        R3: argmax_s sec(s) s ∈ SRC (bad container allowed)
        R4: any empty / rolled stack (last resort)
    """
    excluded_actions = excluded_actions or set()
    src_key = env._idx_to_stack(src_action)

    def _valid(action_idx: int, key) -> bool:
        if action_idx == src_action or action_idx in excluded_actions:
            return False
        stk = env.yard.stacks.get(key)
        return stk is not None and not stk.is_full

    srcs = [(i, k) for i, k in src_stacks(env) if _valid(i, k)]

    # R1
    r1: List[Tuple[int, int, Tuple[int, int]]] = []   # (-se, i, key)
    for i, k in srcs:
        stk = env.yard.stacks[k]
        if is_bad_if_placed_on(blocker, stk):
            continue
        if use_deadlock and creates_deadlock_if_placed(
                env, blocker.group, blocker.priority, k, bad_map):
            continue
        r1.append((-se_of(stk, env._vessel_loaded), i, k))
    if r1:
        return _pick_scored(r1, rng)

    # R2
    r2: List[Tuple[int, int, Tuple[int, int]]] = []
    for i, k in srcs:
        stk = env.yard.stacks[k]
        if is_bad_if_placed_on(blocker, stk):
            continue
        r2.append((-se_of(stk, env._vessel_loaded), i, k))
    if r2:
        return _pick_scored(r2, rng)

    # R3
    r3: List[Tuple[int, int, Tuple[int, int]]] = []
    for i, k in srcs:
        stk = env.yard.stacks[k]
        r3.append((-sec_of(stk, blocker.group, env._vessel_loaded), i, k))
    if r3:
        return _pick_scored(r3, rng)

    # R4 — empty / rolled stack fallback
    fallback = [(i, k) for i, k in empty_or_rolled_stacks(env)
                if _valid(i, k)]
    if fallback:
        return _pick(fallback, rng)

    # Nothing valid excluding forbidden — fall back to any non-full non-src.
    n = env._n_stacks
    any_dsts: List[Tuple[int, Tuple[int, int]]] = []
    for i in range(n):
        if i == src_action:
            continue
        key = env._idx_to_stack(i)
        stk = env.yard.stacks.get(key)
        if stk is not None and not stk.is_full:
            any_dsts.append((i, key))
    if any_dsts:
        return _pick(any_dsts, rng)
    return src_action   # yard is full – no legal move


def _pick(pool: List[Tuple[int, Tuple[int, int]]], rng=None) -> int:
    """Randomly (or deterministically) pick from a list of (action_idx, key)."""
    if not pool:
        return -1
    if rng is None:
        pool_sorted = sorted(pool, key=lambda x: x[1])
        return int(pool_sorted[0][0])
    return int(pool[int(rng.randint(0, len(pool)))][0])


def _pick_scored(
    pool: List[Tuple[int, int, Tuple[int, int]]],
    rng = None,
) -> int:
    """Pick from (score, action_idx, key) — randomize among ties on score."""
    if not pool:
        return -1
    pool.sort()
    best = pool[0][0]
    tied = [(i, k) for score, i, k in pool if score == best]
    return _pick(tied, rng)


# ================================================================ #
#  Episode runner                                                  #
# ================================================================ #

def _next_priority_after_target(env, target_cont) -> Optional[int]:
    """Priority of the OC of same group that will become the next candidate
    after target_cont is retrieved.  None if none exists.
    """
    if target_cont is None:
        return None
    g = target_cont.group
    candidates = [
        c.priority
        for stk in env.yard.stacks.values()
        for c in stk.containers
        if (not is_rc(c)) and c.group == g and c.priority > target_cont.priority
    ]
    if not candidates:
        return None
    return min(candidates)


def _step_high_action(env, action: int) -> Optional:
    """Return the target Container that env._step_high would target for this stack."""
    src_key = env._idx_to_stack(action)
    stk = env.yard.stacks.get(src_key)
    if stk is None or stk.is_empty:
        return None
    for c in reversed(stk.containers):
        if env._is_retrievable(c):
            return c
    return None


def _run_episode(
    env,
    rng           = None,
    use_deadlock  = True,
) -> Tuple[List[int], List[Move], Dict]:
    """
    Run one greedy (deterministic or randomized) episode until termination.
    Records every env.step call.  Returns (actions, records, metrics).
    """
    env.reset()
    auto_retrieve(env)

    actions: List[int] = []
    records: List[Move] = []
    N = len(env.containers)
    max_steps = env._n_stacks * getattr(env.config, "max_tiers", 10) * 6 + 200

    reloc_cnt: Dict[int, int] = defaultdict(int)
    current_target: Optional = None
    high_idx = -1

    while not env._done and len(actions) < max_steps:
        mask = env._build_action_mask()
        if not mask.any():
            break

        if env._mode == "high":
            bad_map = compute_bad_map(env)
            action = _select_target(env, bad_map, rng, use_deadlock)
            if action < 0:
                break
            target = _step_high_action(env, action)
            current_target = target
            high_idx += 1
            rec = Move(
                kind="high",
                action=action,
                cont_id=target.id if target else -1,
                src_key=env._idx_to_stack(action),
                target_id=target.id if target else -1,
                block_idx=high_idx,
            )
        else:
            src_action = env._source_stack_idx
            src_key    = env._idx_to_stack(src_action)
            src_stk    = env.yard.stacks.get(src_key)
            blocker    = src_stk.top if (src_stk and not src_stk.is_empty) else None

            if blocker is None:
                break

            excluded: Set[int] = {src_action}
            if is_rc(blocker):
                next_pri = _next_priority_after_target(env, current_target)
                grp = current_target.group if current_target is not None else -1
                action = _rr_task1_rc(env, src_action, grp, next_pri,
                                      excluded_actions=None, rng=rng)
            else:
                bad_map = compute_bad_map(env)
                action = _rr_task2_oc(env, blocker, src_action, bad_map,
                                      excluded_actions=None,
                                      use_deadlock=use_deadlock, rng=rng)

            dst_key = env._idx_to_stack(action)
            dst_stk = env.yard.stacks.get(dst_key)
            is_bad  = (
                False if is_rc(blocker)
                else is_bad_if_placed_on(blocker, dst_stk) if dst_stk else False
            )
            reloc_cnt[id(blocker)] += 1
            rec = Move(
                kind="low",
                action=action,
                cont_id=blocker.id,
                src_key=src_key,
                dst_key=dst_key,
                is_bad=is_bad,
                reloc_n=reloc_cnt[id(blocker)],
                target_id=current_target.id if current_target is not None else -1,
                block_idx=high_idx,
            )

        _, _, terminated, truncated, _ = env.step(action)
        actions.append(action)
        records.append(rec)
        auto_retrieve(env)
        if terminated or truncated:
            break

    return actions, records, env.get_metrics()


# ================================================================ #
#  Public entry points                                             #
# ================================================================ #

def run_greedy(env, rng=None) -> Tuple[List[int], List[Move], Dict]:
    """Deterministic or randomized greedy (§5.2 / §5.3.2)."""
    return _run_episode(env, rng=rng, use_deadlock=True)


def run_greedy_no_deadlock(env, rng=None) -> Tuple[List[int], List[Move], Dict]:
    """Greedy_Randomized_noRRDL (§5.3.3) — skips deadlock checks for speed."""
    return _run_episode(env, rng=rng, use_deadlock=False)


def replay_actions(env, actions: Sequence[int]) -> Tuple[List[Move], Dict]:
    """Replay an action sequence, rebuilding Move records for LNS analysis."""
    env.reset()
    auto_retrieve(env)
    records: List[Move] = []
    reloc_cnt: Dict[int, int] = defaultdict(int)
    current_target: Optional = None
    high_idx = -1

    for a in actions:
        if env._done:
            break
        mask = env._build_action_mask()
        if not mask.any():
            break

        if env._mode == "high":
            target = _step_high_action(env, a)
            current_target = target
            high_idx += 1
            rec = Move(
                kind="high",
                action=a,
                cont_id=target.id if target else -1,
                src_key=env._idx_to_stack(a),
                target_id=target.id if target else -1,
                block_idx=high_idx,
            )
        else:
            src_action = env._source_stack_idx
            src_key    = env._idx_to_stack(src_action)
            src_stk    = env.yard.stacks.get(src_key)
            blocker    = src_stk.top if (src_stk and not src_stk.is_empty) else None
            dst_key    = env._idx_to_stack(a)
            dst_stk    = env.yard.stacks.get(dst_key)
            is_bad = (
                False if (blocker is None or is_rc(blocker))
                else is_bad_if_placed_on(blocker, dst_stk) if dst_stk else False
            )
            if blocker is not None:
                reloc_cnt[id(blocker)] += 1
            rec = Move(
                kind="low",
                action=a,
                cont_id=blocker.id if blocker else -1,
                src_key=src_key,
                dst_key=dst_key,
                is_bad=is_bad,
                reloc_n=reloc_cnt[id(blocker)] if blocker else 0,
                target_id=current_target.id if current_target is not None else -1,
                block_idx=high_idx,
            )
        env.step(a)
        records.append(rec)
        auto_retrieve(env)

    return records, env.get_metrics()
