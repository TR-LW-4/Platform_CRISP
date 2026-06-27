"""
Zeng et al. (2019) – Five heuristics (H_1 to H_5) for CRP-D.

Reference
---------
Zeng, Q., Feng, Y., & Yang, Z. (2019). "Integrated optimization of pickup
sequence and container rehandling based on partial truck arrival information."
Computers & Industrial Engineering, 127, 366–382.

Problem correspondence
----------------------
- The paper's "groups" correspond to CRP-D duplicate-priority groups.
- Group priorities are ordered: group 1 before group 2, etc.
- Within a group the retrieval order is free (the paper optimises this).
- Compatible with the CRP-D platform only.
- Action space: Discrete(n²), action = src_idx * n + dst_idx.

Algorithm structure (shared by all five heuristics)
---------------------------------------------------
At every step the CRP-D environment has already auto-retrieved all accessible
target-group containers.  The algorithm must issue one relocation action.

Source selection (§4.2.1 — "sequencing rule"):
    Among stacks that contain at least one blocked target-group container:
    1. Pick the stack with LARGEST ḡ_min(s, k)
       (the non-target containers in that stack will be retrieved later → safer
       to disturb first).
    2. Tie-break: smallest (bay, row).

Destination selection (§4.2.2 — "relocation rule"):
    The five heuristics share the same priority tiers for the destination:

    Priority 1 — "first-kind" stacks: non-empty, gmin(s) > k
                  (all containers retrieved AFTER current group; safest).
                  Among these → smallest gmin(s) closest to k.
    Priority 2 — empty stacks  (tie-break: smallest (bay, row)).
    Priority 3 — "sub-first-kind": gmin(s) == k  (tie-break: smallest (bay, row)).
    Fallback    — remaining stacks where gmin(s) < k:

        H_1  (H6 rule)      → smallest BI,  tie-break: largest gmin(s)
        H_2  (Adj-H1 rule)  → smallest RI,  tie-break: largest gmin(s)
        H_3  (Adj-H2 rule)  → smallest BI,  tie-break: smallest RI

For H_4 (Adj-H4) and H_5 (Adj-H5):
    When a source is chosen, pre-assign destinations for ALL blockers above the
    highest blocked target-group container in that stack, sorted by DECREASING
    group code (later-retrieved blockers assigned first to minimise cascading
    rehandles).  The physical relocation plan is then executed top-to-bottom.
    Uses "adjusted RI / BI" accounting for virtual placements in each round.

Helper definitions
------------------
    k          : current target group code
    gmin(s)    : min group code among containers in stack s
    ḡmin(s, k) : min group code in s, excluding group k
    RI(s, g_b) : number of containers in s with group < g_b
    BI(s)      : containers above the earliest-to-retrieve container in s,
                 after placing one more container on top
"""

from __future__ import annotations

import multiprocessing as mp
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig


# ================================================================ #
#  Tiny env-level helpers                                          #
# ================================================================ #

def _n_stacks(env) -> int:
    return env.config.num_bays * env.config.num_rows


def _target_group(env) -> int:
    tg = env._current_target_group()
    return tg if tg is not None else 0


# ================================================================ #
#  Stack-level helper functions                                    #
# ================================================================ #

def _gmin(stk) -> float:
    """Minimum group code in stack (inf if empty)."""
    if stk is None or stk.is_empty:
        return float("inf")
    return min(c.group for c in stk.containers)


def _gbar_min(stk, k: int) -> float:
    """Minimum group code in stack, excluding group k (inf if none)."""
    if stk is None or stk.is_empty:
        return float("inf")
    codes = [c.group for c in stk.containers if c.group != k]
    return min(codes) if codes else float("inf")


def _RI(stk, g_b: int, extra_groups: Optional[List[int]] = None) -> int:
    """
    Relocation Interruptions: count containers in stk (plus virtual extras)
    with group code < g_b.
    """
    count = sum(1 for c in stk.containers if c.group < g_b)
    if extra_groups:
        count += sum(1 for g in extra_groups if g < g_b)
    return count


def _BI(stk, extra_groups: Optional[List[int]] = None) -> int:
    """
    Blocking Index: after placing ONE more container on top of stk, how many
    containers sit above the earliest-to-retrieve (gmin) container?
    """
    real      = [c.group for c in stk.containers] if (stk and not stk.is_empty) else []
    virtual   = list(extra_groups) if extra_groups else []
    effective = real + virtual

    if not effective:
        return 0

    gmin     = min(effective)
    idx_gmin = max(i for i, g in enumerate(effective) if g == gmin)
    return len(effective) - idx_gmin


# ================================================================ #
#  Source selection (replaces two-phase "high" selector)          #
# ================================================================ #

def _select_source(env, k: int) -> int:
    """
    Pick the source stack index to work on next.

    Among stacks whose top is NOT a target-group container and that contain
    at least one blocked group-k container:
      → pick the one with LARGEST ḡ_min(s, k) (moving containers that will be
        retrieved late minimises future ripple).  Tie-break: smallest (bay, row).

    Fallback (no stack contains a group-k container):
      → any non-empty stack whose top is relocatable (not k), smallest (bay, row).
    """
    n          = _n_stacks(env)
    candidates: List[Tuple] = []
    fallback:   List[Tuple] = []

    for i in range(n):
        key = env._idx_to_stack(i)
        stk = env.yard.stacks.get(key)
        if stk is None or stk.is_empty:
            continue
        if stk.top.group == k:
            continue  # accessible target – already auto-retrieved normally

        has_k = any(c.group == k for c in stk.containers)
        if has_k:
            gbar = _gbar_min(stk, k)
            candidates.append((-gbar, key[0], key[1], i))
        else:
            fallback.append((key[0], key[1], i))

    if candidates:
        candidates.sort()
        return candidates[0][3]
    if fallback:
        fallback.sort()
        return fallback[0][2]
    return 0


# ================================================================ #
#  Destination-stack enumeration                                   #
# ================================================================ #

def _iter_destinations(env, src_key: Tuple[int, int], k: int):
    """
    Yield candidate destination stacks (not src_key, not full) classified into:
      ("first",  gmin,  bay, row, dst_idx)  — gmin > k
      ("empty",  inf,   bay, row, dst_idx)
      ("sub",    k,     bay, row, dst_idx)  — gmin == k
      ("last",   gmin,  bay, row, dst_idx)  — gmin < k
    """
    n = _n_stacks(env)
    for i in range(n):
        key = env._idx_to_stack(i)
        if key == src_key:
            continue
        stk = env.yard.stacks.get(key)
        if stk is None or stk.is_full:
            continue
        bay, row = key
        if stk.is_empty:
            yield ("empty", float("inf"), bay, row, i, key)
        else:
            gm = _gmin(stk)
            if gm > k:
                yield ("first", gm, bay, row, i, key)
            elif gm == k:
                yield ("sub", gm, bay, row, i, key)
            else:
                yield ("last", gm, bay, row, i, key)


# ================================================================ #
#  Relocation rules                                                #
# ================================================================ #

def _relocate_H6(env, src_key: Tuple[int, int], k: int,
                 extra: Optional[Dict[Tuple, List[int]]] = None) -> int:
    """
    H6 rule (used by H_1):
      P1: first-kind (gmin > k)   → smallest gmin, tie-break (bay, row)
      P2: empty stacks            → smallest (bay, row)
      P3: sub-first-kind (gmin=k) → smallest (bay, row)
      Fallback (gmin < k)         → largest gmin (closest to k), tie-break smallest BI
    """
    first: List[Tuple] = []
    empty: List[Tuple] = []
    sub:   List[Tuple] = []
    last:  List[Tuple] = []

    for tier, gm, bay, row, act, key in _iter_destinations(env, src_key, k):
        stk_extra = (extra or {}).get(key, [])
        if tier == "first":
            first.append((gm, bay, row, act))
        elif tier == "empty":
            empty.append((bay, row, act))
        elif tier == "sub":
            sub.append((bay, row, act))
        else:
            stk = env.yard.stacks[key]
            bi  = _BI(stk, stk_extra)
            last.append((-gm, bi, bay, row, act))

    if first: first.sort(); return first[0][3]
    if empty: empty.sort(); return empty[0][2]
    if sub:   sub.sort();   return sub[0][2]
    if last:  last.sort();  return last[0][4]
    return 0


def _relocate_AH1(env, src_key: Tuple[int, int], k: int,
                  blocker_group: int,
                  extra: Optional[Dict[Tuple, List[int]]] = None) -> int:
    """
    Adjusted-H1 rule (used by H_2):
      P1: first-kind  → smallest gmin
      P2: empty
      P3: sub-first-kind
      Fallback: smallest RI, tie-break largest gmin
    """
    first: List[Tuple] = []
    empty: List[Tuple] = []
    sub:   List[Tuple] = []
    last:  List[Tuple] = []

    for tier, gm, bay, row, act, key in _iter_destinations(env, src_key, k):
        stk_extra = (extra or {}).get(key, [])
        if tier == "first":
            first.append((gm, bay, row, act))
        elif tier == "empty":
            empty.append((bay, row, act))
        elif tier == "sub":
            sub.append((bay, row, act))
        else:
            stk = env.yard.stacks[key]
            ri  = _RI(stk, blocker_group, stk_extra)
            last.append((ri, -gm, bay, row, act))

    if first: first.sort(); return first[0][3]
    if empty: empty.sort(); return empty[0][2]
    if sub:   sub.sort();   return sub[0][2]
    if last:  last.sort();  return last[0][4]
    return 0


def _relocate_AH2(env, src_key: Tuple[int, int], k: int,
                  blocker_group: int,
                  extra: Optional[Dict[Tuple, List[int]]] = None) -> int:
    """
    Adjusted-H2 rule (used by H_3):
      P1: first-kind  → smallest gmin
      P2: empty
      P3: sub-first-kind
      Fallback: smallest BI, tie-break smallest RI
    """
    first: List[Tuple] = []
    empty: List[Tuple] = []
    sub:   List[Tuple] = []
    last:  List[Tuple] = []

    for tier, gm, bay, row, act, key in _iter_destinations(env, src_key, k):
        stk_extra = (extra or {}).get(key, [])
        if tier == "first":
            first.append((gm, bay, row, act))
        elif tier == "empty":
            empty.append((bay, row, act))
        elif tier == "sub":
            sub.append((bay, row, act))
        else:
            stk = env.yard.stacks[key]
            bi  = _BI(stk, stk_extra)
            ri  = _RI(stk, blocker_group, stk_extra)
            last.append((bi, ri, bay, row, act))

    if first: first.sort(); return first[0][3]
    if empty: empty.sort(); return empty[0][2]
    if sub:   sub.sort();   return sub[0][2]
    if last:  last.sort();  return last[0][4]
    return 0


# ================================================================ #
#  Pre-computation for H_4 / H_5                                  #
# ================================================================ #

def _precompute_plan(env, src_idx: int, k: int, use_bi: bool) -> Deque[int]:
    """
    For H_4 (use_bi=False, adjusted RI) and H_5 (use_bi=True, adjusted BI).

    Identify all blockers above the topmost group-k container in the source
    stack, sort by DECREASING group code, assign destinations greedily using
    virtual accounting, then return a deque of destination indices ordered
    top-to-bottom (physical execution order).
    """
    src_key = env._idx_to_stack(src_idx)
    src_stk = env.yard.stacks.get(src_key)

    if src_stk is None or src_stk.is_empty:
        return deque()

    # Blockers: from top down until we hit a group-k container
    containers_top_down = list(reversed(src_stk.containers))
    blockers: List[Tuple[int, int]] = []   # (physical_pos, group_code)
    for phys_pos, c in enumerate(containers_top_down):
        if c.group == k:
            break
        blockers.append((phys_pos, c.group))

    if not blockers:
        return deque()

    # Assign destinations in DECREASING group-code order
    assignment_order = sorted(blockers, key=lambda x: -x[1])
    virtual: Dict[Tuple, List[int]] = {}
    dest_by_phys_pos: Dict[int, int] = {}

    for phys_pos, g_b in assignment_order:
        if use_bi:
            dst = _relocate_AH2(env, src_key, k, g_b, virtual)
        else:
            dst = _relocate_AH1(env, src_key, k, g_b, virtual)
        dest_key = env._idx_to_stack(dst)
        virtual.setdefault(dest_key, []).append(g_b)
        dest_by_phys_pos[phys_pos] = dst

    # Return in physical top-to-bottom order
    plan: Deque[int] = deque()
    for phys_pos, _ in sorted(blockers, key=lambda x: x[0]):
        plan.append(dest_by_phys_pos[phys_pos])
    return plan


# ================================================================ #
#  Episode runner                                                  #
# ================================================================ #

def _run_episode(env, selector_fn: Callable) -> Tuple[List[int], Dict]:
    """
    Run one complete episode using selector_fn(env, state) → action.
    selector_fn receives a mutable `state` dict for per-episode storage.
    """
    env.reset()
    solution: List[int] = []
    done  = False
    state: Dict = {}
    n     = _n_stacks(env)
    max_steps = n * getattr(env.config, "max_tiers", 10) * 4 + 200

    while not done and len(solution) < max_steps:
        mask = env._build_action_mask() if hasattr(env, "_build_action_mask") else None
        if mask is not None and not mask.any():
            break

        action = selector_fn(env, state)
        _, _, terminated, truncated, _ = env.step(action)
        solution.append(action)
        done = terminated or truncated

    return solution, env.get_metrics()


# ================================================================ #
#  Base algorithm class                                            #
# ================================================================ #

class _ZengBase(BaseAlgorithm):
    """Base class for Zeng et al. (2019) heuristics (CRP-D)."""

    compatible_problems = ["CRP-D"]
    step_label          = "Seed"

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    def _select(self, env, state: Dict) -> int:
        raise NotImplementedError

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        cfg     = self.config
        n_seeds = max(1, cfg.num_eval_seeds)
        all_metrics: List[Dict] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.config.seed = cfg.seed + seed

            solution, metrics = _run_episode(env, self._select)
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

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "num_eval_seeds": {
                "type": "int", "default": 10, "min": 1, "max": 200,
                "label": "Evaluation seeds",
            },
        })
        return base


# ================================================================ #
#  H_1  — uses H6 relocation rule                                 #
# ================================================================ #

class ZengH1(_ZengBase):

    name        = "Zeng (2019) H₁"
    category    = "Heuristic"
    description = (
        "Zeng et al. (2019) H_1 heuristic for CRP-D. "
        "Source: stack with largest ḡ_min among blocked-target stacks. "
        "Relocation: H6 — first-kind stacks first (gmin > k, smallest gmin), "
        "then empty, then sub-first-kind (gmin=k), finally smallest BI."
    )

    def _select(self, env, state: Dict) -> int:
        n       = _n_stacks(env)
        k       = _target_group(env)
        src     = _select_source(env, k)
        src_key = env._idx_to_stack(src)
        dst     = _relocate_H6(env, src_key, k)
        return src * n + dst


# ================================================================ #
#  H_2  — uses Adjusted-H1 relocation rule (min RI fallback)     #
# ================================================================ #

class ZengH2(_ZengBase):

    name        = "Zeng (2019) H₂"
    category    = "Heuristic"
    description = (
        "Zeng et al. (2019) H_2 heuristic for CRP-D. "
        "Same source selection as H_1. "
        "Relocation: Adj-H1 — same priorities as H6 but fallback uses "
        "smallest RI (Relocation Interruptions) instead of BI."
    )

    def _select(self, env, state: Dict) -> int:
        n       = _n_stacks(env)
        k       = _target_group(env)
        src     = _select_source(env, k)
        src_key = env._idx_to_stack(src)
        src_stk = env.yard.stacks.get(src_key)
        g_b     = src_stk.top.group if (src_stk and not src_stk.is_empty) else 0
        dst     = _relocate_AH1(env, src_key, k, g_b)
        return src * n + dst


# ================================================================ #
#  H_3  — uses Adjusted-H2 relocation rule (min BI, tie RI)      #
# ================================================================ #

class ZengH3(_ZengBase):

    name        = "Zeng (2019) H₃"
    category    = "Heuristic"
    description = (
        "Zeng et al. (2019) H_3 heuristic for CRP-D. "
        "Same source selection as H_1. "
        "Relocation: Adj-H2 — same priorities as H6 but fallback uses "
        "smallest BI with smallest RI as tie-break."
    )

    def _select(self, env, state: Dict) -> int:
        n       = _n_stacks(env)
        k       = _target_group(env)
        src     = _select_source(env, k)
        src_key = env._idx_to_stack(src)
        src_stk = env.yard.stacks.get(src_key)
        g_b     = src_stk.top.group if (src_stk and not src_stk.is_empty) else 0
        dst     = _relocate_AH2(env, src_key, k, g_b)
        return src * n + dst


# ================================================================ #
#  H_4  — Adjusted-H4: sorted reloc order + adjusted RI          #
# ================================================================ #

class ZengH4(_ZengBase):

    name        = "Zeng (2019) H₄"
    category    = "Heuristic"
    description = (
        "Zeng et al. (2019) H_4 heuristic for CRP-D. "
        "Same source selection as H_1. "
        "Relocation: Adj-H4 — pre-assign destinations for ALL blockers above "
        "the target, sorted by DECREASING group code, using adjusted RI."
    )

    def _select(self, env, state: Dict) -> int:
        n   = _n_stacks(env)
        k   = _target_group(env)
        src = _select_source(env, k)

        if state.get("src") != src or not state.get("plan"):
            state["src"]  = src
            state["plan"] = _precompute_plan(env, src, k, use_bi=False)

        src_key = env._idx_to_stack(src)
        plan: Deque[int] = state["plan"]
        if plan:
            dst     = plan.popleft()
            dst_key = env._idx_to_stack(dst)
            dst_stk = env.yard.stacks.get(dst_key)
            if dst_key != src_key and dst_stk and not dst_stk.is_full:
                return src * n + dst

        # Fallback: Adj-H1 on current top
        src_stk = env.yard.stacks.get(src_key)
        g_b     = src_stk.top.group if (src_stk and not src_stk.is_empty) else 0
        dst     = _relocate_AH1(env, src_key, k, g_b)
        return src * n + dst


# ================================================================ #
#  H_5  — Adjusted-H5: sorted reloc order + adjusted BI          #
# ================================================================ #

class ZengH5(_ZengBase):

    name        = "Zeng (2019) H₅"
    category    = "Heuristic"
    description = (
        "Zeng et al. (2019) H_5 heuristic for CRP-D. "
        "Same source selection as H_1. "
        "Relocation: Adj-H5 — same pre-sorted assignment as H_4 but uses "
        "adjusted BI (Blocking Index) instead of adjusted RI."
    )

    def _select(self, env, state: Dict) -> int:
        n   = _n_stacks(env)
        k   = _target_group(env)
        src = _select_source(env, k)

        if state.get("src") != src or not state.get("plan"):
            state["src"]  = src
            state["plan"] = _precompute_plan(env, src, k, use_bi=True)

        src_key = env._idx_to_stack(src)
        plan: Deque[int] = state["plan"]
        if plan:
            dst     = plan.popleft()
            dst_key = env._idx_to_stack(dst)
            dst_stk = env.yard.stacks.get(dst_key)
            if dst_key != src_key and dst_stk and not dst_stk.is_full:
                return src * n + dst

        # Fallback: Adj-H2 on current top
        src_stk = env.yard.stacks.get(src_key)
        g_b     = src_stk.top.group if (src_stk and not src_stk.is_empty) else 0
        dst     = _relocate_AH2(env, src_key, k, g_b)
        return src * n + dst
