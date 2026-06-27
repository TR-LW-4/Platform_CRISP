"""
Firmino, Silva & Times (2019) — Reactive GRASP for CRP-Time.

Reference
---------
A. da Silva Firmino, R. M. de Abreu Silva, V. C. Times,
"A reactive GRASP metaheuristic for the container retrieval problem to
 reduce crane's working time",
Journal of Heuristics 25 (2019) 141–173.
https://doi.org/10.1007/s10732-018-9390-0

Algorithm overview
------------------
Each GRASP iteration consists of two phases:

1. **Construction Phase** (Algorithm 2 in paper):
   A semi-greedy heuristic that, at each step, builds a Restricted
   Candidate List (RCL) using the MNI decision index (Jovanovic & Voß
   2014) and randomly selects a relocation from the RCL.

   MNI score for placing container *c* into stack *s* with min-priority *vs*:
     score = vs − c           if vs > c   (no blocking, lower = better fit)
     score = 2N + 1 − vs      if vs ≤ c   (blocking)
     +extra penalty            if the stack would become full AND creates blocking

   RCL greedy rate α ∈ [0, 1]:
     GLimitα = max_score − α · (max_score − min_score)
     RLC = {s | MNI(s) ≤ GLimitα}
   When α = 0: full random (all candidates). α = 1: pure greedy.

2. **Local Search Phase** (Algorithm 3 in paper):
   Scans moves in reverse order. For each relocation, "undoes" it and
   attempts to find a substitute destination that:
     (a) has not been visited in this pass,
     (b) produces no blocking (min_prio > c), and
     (c) reduces the total crane working time.
   When a valid replacement is found, the subsequent portion of the
   solution is rebuilt by re-running the greedy MNI heuristic from the
   new yard state.

3. **Reactive α** (paper §7):
   α is adjusted every iteration based on whether the new solution
   improved on the previous one:
     • improved → continue the current adjustment direction
     • worsened → reverse the adjustment direction
   Start: α = 1.0 (greedy), step Δv = 0.05, direction D (decrement).

Crane-time model
----------------
Uses the platform's Lee & Lee (2010) kinematics (`compute_crane_time`)
to ensure that the reported `crane_time` is directly comparable with
all other CRP-Time algorithms on the same benchmarks.

The paper's own 4-speed model (ν₁–ν₄, horizontal + vertical) is NOT
replicated here because: (a) it is a different physical model and makes
results incomparable, and (b) the platform's model is the canonical one
for the Lee_instances and Shin_instances benchmarks.

Bay support
-----------
[single-bay origin]  The paper restricts all relocations to the same bay
(no cross-bay moves, as stated in Sect. 1).  On single-bay instances
(num_bays = 1) the algorithm is fully faithful to the paper.  On
multi-bay instances the greedy selection considers all valid stacks across
bays, but the MNI index does NOT penalise cross-bay gantry travel, so
the behaviour is analogous to GLAH in this platform: crane-time is
correctly computed but the heuristic decision ignores bay distance.
"""

from __future__ import annotations

import multiprocessing as mp
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.objectives import KinematicsModel, compute_crane_time
from core.plan import Movement, RelocationPlan


# ================================================================ #
#  Type aliases                                                      #
# ================================================================ #

# Internal yard: (bay, row) → list of container priorities bottom-to-top
Stacks = Dict[Tuple[int, int], List[int]]
# A single relocation: (container_priority, src_key, dst_key)
# retrieval:            (container_priority, src_key, None)
Step = Tuple[int, Any, Any]


# ================================================================ #
#  MNI decision index                                                #
# ================================================================ #

def _mni_score(
    c:         int,
    dst_prios: List[int],
    n_total:   int,
    max_tiers: int,
) -> float:
    """
    MNI score for placing container *c* into a stack whose current
    priorities are *dst_prios*.  Lower score = better destination.

    Follows the Min-Max Index (Jovanovic & Voß 2014) as described in
    Firmino et al. (2019) Sect. 2:
      vs  = min priority in the destination stack (N+1 if empty)
      vs* = -(N + vs)  when the stack would become full AND vs < c
            else  vs  (standard)

    We convert to an ascending "badness" score:
      badness = vs - c         if vs > c  (no blocking; smaller = better)
      badness = 2N + 1 - vs    if vs ≤ c  (blocking; varies)
      +  (3N + 2)              if MNI fill-penalty applies
    """
    if not dst_prios:
        vs = n_total + 1      # empty stack: never blocks anything
    else:
        vs = min(dst_prios)

    # MNI penalty: stack would be full after this move AND creates blocking
    will_fill = (len(dst_prios) + 1) >= max_tiers
    mni_penalty = will_fill and (vs < c)

    if vs > c:
        base = float(vs - c)  # no blocking; lower = closer fit
    else:
        base = float(2 * n_total + 1 - vs)  # blocking

    return base + (3 * n_total + 2 if mni_penalty else 0.0)


# ================================================================ #
#  Crane-time from abstract move list                                #
# ================================================================ #

def _plan_from_moves(
    moves:       List[Step],
    stacks_init: Stacks,
    n_total:     int,
) -> RelocationPlan:
    """
    Replay the move list on a local copy of the initial yard and build
    a `RelocationPlan` (which includes both relocations and retrievals).
    """
    stacks: Stacks = {k: list(v) for k, v in stacks_init.items()}
    loc: Dict[int, Any] = {}
    for key, prios in stacks_init.items():
        for p in prios:
            loc[p] = key

    plan = RelocationPlan()
    move_ptr = 0
    cur_target = 1

    while cur_target <= n_total:
        src_key = loc.get(cur_target)
        if src_key is None:
            cur_target += 1
            continue

        # Relocate blockers above cur_target
        while stacks[src_key] and stacks[src_key][-1] != cur_target:
            if move_ptr >= len(moves):
                break
            c, _src, dst_key = moves[move_ptr]
            move_ptr += 1
            stacks[src_key].pop()
            stacks[dst_key].append(c)
            loc[c] = dst_key
            plan.add(Movement(c, src_key, dst_key))

        # Retrieve cur_target
        if stacks[src_key] and stacks[src_key][-1] == cur_target:
            stacks[src_key].pop()
            plan.add(Movement(cur_target, src_key, None))
            loc[cur_target] = None

        cur_target += 1

    return plan


def _crane_time_from_moves(
    moves:       List[Step],
    stacks_init: Stacks,
    n_total:     int,
    kin:         KinematicsModel,
) -> float:
    return float(compute_crane_time(_plan_from_moves(moves, stacks_init, n_total), kin))


# ================================================================ #
#  Single-step greedy selector (MNI, deterministic)                  #
# ================================================================ #

def _greedy_select(
    c:         int,
    src_key:   Any,
    stacks:    Stacks,
    all_keys:  List[Any],
    n_total:   int,
    max_tiers: int,
) -> Optional[Any]:
    """Return the dst_key with lowest MNI score (ties: first found)."""
    best_dst   = None
    best_score = float("inf")
    for dst_key in all_keys:
        if dst_key == src_key:
            continue
        prios = stacks[dst_key]
        if len(prios) >= max_tiers:
            continue
        score = _mni_score(c, prios, n_total, max_tiers)
        if score < best_score:
            best_score = score
            best_dst   = dst_key
    return best_dst


# ================================================================ #
#  Construction phase (Algorithm 2)                                  #
# ================================================================ #

def _construction_phase(
    stacks_init: Stacks,
    all_keys:    List[Any],
    n_total:     int,
    max_tiers:   int,
    alpha:       float,
    rng:         random.Random,
) -> List[Step]:
    """
    Semi-greedy construction using the MNI index and adaptive RCL.

    Returns a list of (c, src_key, dst_key) relocation steps.
    Retrieval steps are implicit (handled during replay).
    """
    stacks: Stacks = {k: list(v) for k, v in stacks_init.items()}
    loc: Dict[int, Any] = {}
    for key, prios in stacks_init.items():
        for p in prios:
            loc[p] = key

    moves: List[Step] = []

    for target in range(1, n_total + 1):
        src_key = loc.get(target)
        if src_key is None:
            continue

        while stacks[src_key] and stacks[src_key][-1] != target:
            c = stacks[src_key][-1]

            # Compute MNI score for each candidate destination
            cands: List[Tuple[float, Any]] = []
            for dst_key in all_keys:
                if dst_key == src_key:
                    continue
                if len(stacks[dst_key]) >= max_tiers:
                    continue
                score = _mni_score(c, stacks[dst_key], n_total, max_tiers)
                cands.append((score, dst_key))

            if not cands:
                break  # no valid destination (pathological)

            scores    = [s for s, _ in cands]
            min_score = min(scores)
            max_score = max(scores)

            # Greedy limit: lower α → tighter limit → more random
            # α=1: only best; α=0: all candidates
            g_limit = max_score - alpha * (max_score - min_score)
            rlc = [dst for s, dst in cands if s <= g_limit + 1e-9]
            if not rlc:
                rlc = [cands[0][1]]  # fallback: take best

            dst_key = rng.choice(rlc)

            moves.append((c, src_key, dst_key))
            stacks[src_key].pop()
            stacks[dst_key].append(c)
            loc[c] = dst_key

        # Retrieve target
        if stacks[src_key] and stacks[src_key][-1] == target:
            stacks[src_key].pop()
            loc[target] = None

    return moves


# ================================================================ #
#  Local search phase (Algorithm 3)                                  #
# ================================================================ #

def _local_search_phase(
    moves:       List[Step],
    stacks_init: Stacks,
    all_keys:    List[Any],
    n_total:     int,
    max_tiers:   int,
    kin:         KinematicsModel,
    current_ct:  float,
) -> Tuple[List[Step], float]:
    """
    Reverse-scan local search (Algorithm 3 from Firmino et al. 2019).

    For each relocation move (in reverse order):
      1. Undo the move (restore the yard to the state before it).
      2. Search for an alternative destination `new_dst` that:
           a. Has not been 'visited' in this pass.
           b. Creates no blocking: min_prio(new_dst) > c (or empty stack).
           c. After substitution + greedy rebuild, gives lower crane_time.
      3. If found: replace the move and greedily rebuild the suffix.
      4. Mark the chosen dst (old or new) as visited.

    Returns (best_moves, best_crane_time).
    """
    best_moves = list(moves)
    best_ct    = current_ct
    visited: set = set()

    for i in range(len(moves) - 1, -1, -1):
        c, old_src, old_dst = best_moves[i]

        # Undo: find yard state at position i by replaying [0..i-1]
        stacks_at_i: Stacks = {k: list(v) for k, v in stacks_init.items()}
        loc_at_i: Dict[int, Any] = {}
        for key, prios in stacks_init.items():
            for p in prios:
                loc_at_i[p] = key

        cur_t = 1
        for j in range(i):
            cj, _srcj, dstj = best_moves[j]
            src_j = loc_at_i[cj]
            stacks_at_i[src_j].pop()
            stacks_at_i[dstj].append(cj)
            loc_at_i[cj] = dstj

            # Auto-retrieve containers that become accessible
            while cur_t <= n_total:
                tk = loc_at_i.get(cur_t)
                if tk and stacks_at_i[tk] and stacks_at_i[tk][-1] == cur_t:
                    stacks_at_i[tk].pop()
                    loc_at_i[cur_t] = None
                    cur_t += 1
                else:
                    break

        # Current src is where c is now (top of src_key that contains target)
        real_src = loc_at_i.get(c)
        if real_src is None:
            visited.add(old_dst)
            continue

        # Try alternative destinations
        improved = False
        for new_dst in all_keys:
            if new_dst == real_src:
                continue
            if new_dst == old_dst:
                continue
            if new_dst in visited:
                continue
            if len(stacks_at_i[new_dst]) >= max_tiers:
                continue

            dst_prios = stacks_at_i[new_dst]
            # Condition: no new blocking
            if dst_prios and min(dst_prios) <= c:
                continue  # would create blocking

            # Apply new_dst move and greedily rebuild the rest
            stacks_trial: Stacks = {k: list(v) for k, v in stacks_at_i.items()}
            loc_trial: Dict[int, Any] = dict(loc_at_i)

            stacks_trial[real_src].pop()
            stacks_trial[new_dst].append(c)
            loc_trial[c] = new_dst

            # Auto-retrieve
            t = cur_t
            while t <= n_total:
                tk = loc_trial.get(t)
                if tk and stacks_trial[tk] and stacks_trial[tk][-1] == t:
                    stacks_trial[tk].pop()
                    loc_trial[t] = None
                    t += 1
                else:
                    break

            # Rebuild remaining moves greedily (deterministic MNI)
            prefix = best_moves[:i] + [(c, real_src, new_dst)]
            suffix = _construction_phase(
                stacks_trial, all_keys, n_total, max_tiers,
                alpha=1.0,  # greedy (deterministic) for LS
                rng=random.Random(0),
            )

            candidate_moves = prefix + suffix
            cand_ct = _crane_time_from_moves(candidate_moves, stacks_init, n_total, kin)

            if cand_ct < best_ct - 1e-6:
                best_moves = candidate_moves
                best_ct    = cand_ct
                visited.add(new_dst)
                improved   = True
                break

        if not improved:
            visited.add(old_dst)

    return best_moves, best_ct


# ================================================================ #
#  Main algorithm class                                              #
# ================================================================ #

class FirminoRGRASP(BaseAlgorithm):

    name     = "Firmino et al. (2019) Reactive GRASP"
    category = "Heuristic"
    description = (
        "[single-bay origin]  "
        "Reactive GRASP for CRP-Time (Firmino, Silva & Times, J. Heuristics 2019). "
        "Construction: semi-greedy MNI decision index with auto-adjusting greedy "
        "rate α (reactive mechanism). "
        "Local search: reverse-scan move substitution that replaces each "
        "relocation with a no-blocking alternative if it lowers crane time, "
        "then greedily rebuilds the suffix. "
        "Primary metric: crane working time (Lee & Lee 2010 kinematics). "
        "On multi-bay instances the MNI index does not penalise cross-bay "
        "travel; use with num_bays=1 for faithful replication of paper results."
    )
    compatible_problems = ["CRP-Time"]
    step_label          = "Iteration"

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    # ---------------------------------------------------------------- #
    # Train                                                              #
    # ---------------------------------------------------------------- #

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        cfg          = self.config
        n_seeds      = max(1, cfg.num_eval_seeds)
        max_iter     = int  (cfg.extra.get("max_iterations",      500))
        max_no_impv  = int  (cfg.extra.get("max_no_improve",    10000))
        alpha_init   = float(cfg.extra.get("alpha_init",          1.0))
        delta_v      = float(cfg.extra.get("delta_v",             0.05))
        do_ls        = bool (cfg.extra.get("local_search",        True))
        report_every = max(1, max_iter // 40)

        all_metrics: List[Dict] = []

        for seed_idx in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.config.seed = cfg.seed + seed_idx
            env.reset()

            n_total   = int(env.config.num_containers)
            max_tiers = int(env.config.max_tiers)
            n_stacks  = env.config.num_bays * env.config.num_rows
            kin       = KinematicsModel.from_config_extra(env.config.extra)

            all_keys:    List[Any] = []
            stacks_init: Stacks   = {}
            for a in range(n_stacks):
                key = env._action_to_stack(a)
                all_keys.append(key)
                stk = env.yard.stacks.get(key)
                stacks_init[key] = (
                    [int(c.priority) for c in stk.containers] if stk else []
                )

            rng        = random.Random(cfg.seed + seed_idx)
            best_ct    = float("inf")
            best_moves: List[Step] = []

            # ── Reactive α state ─────────────────────────────────── #
            alpha          = alpha_init   # 1.0 = greedy
            direction      = -1           # −1 = decreasing (towards random)
            prev_ct        = float("inf")
            no_improve_cnt = 0

            for it in range(1, max_iter + 1):
                if stop_event.is_set():
                    break
                if no_improve_cnt >= max_no_impv:
                    break

                # ── Construction ──────────────────────────────────── #
                moves = _construction_phase(
                    stacks_init, all_keys, n_total, max_tiers, alpha, rng
                )
                ct = _crane_time_from_moves(moves, stacks_init, n_total, kin)

                # ── Local search ──────────────────────────────────── #
                if do_ls and moves:
                    moves, ct = _local_search_phase(
                        moves, stacks_init, all_keys, n_total, max_tiers, kin, ct
                    )

                # ── Update best ───────────────────────────────────── #
                if ct < best_ct:
                    best_ct    = ct
                    best_moves = moves
                    no_improve_cnt = 0
                else:
                    no_improve_cnt += 1

                # ── Reactive α update ─────────────────────────────── #
                if ct < prev_ct:
                    pass  # solution improved → continue current direction
                else:
                    direction = -direction  # solution worsened → reverse direction

                alpha = max(0.0, min(1.0, alpha + direction * delta_v))
                prev_ct = ct

                # ── Periodic report ───────────────────────────────── #
                if it % report_every == 0:
                    frac = (seed_idx * max_iter + it) / (n_seeds * max_iter)
                    relocs = len(best_moves)
                    self._push(
                        result_queue,
                        step     = seed_idx * max_iter + it,
                        metric   = float(best_ct),
                        metrics  = {
                            "crane_time":  float(best_ct),
                            "relocations": float(relocs),
                            "time":        float(best_ct),
                            "alpha":       float(alpha),
                            "iteration":   float(it),
                        },
                        progress = min(frac, (seed_idx + 1) / n_seeds),
                    )

            # ── Seed done: compile plan, get full metrics ─────────── #
            if best_moves:
                plan     = _plan_from_moves(best_moves, stacks_init, n_total)
                ct_final = float(compute_crane_time(plan, kin))
                relocs_final = float(
                    sum(1 for m in plan.movements if not m.is_retrieval)
                )
            else:
                ct_final     = 0.0
                relocs_final = 0.0

            metrics = {
                "crane_time":  ct_final,
                "relocations": relocs_final,
                "time":        ct_final,
                "steps":       relocs_final,
                "progress":    1.0,
            }
            all_metrics.append(metrics)

            if ct_final < self._best_metric:
                self._best_metric   = ct_final
                self._best_solution = []

            self._push(
                result_queue,
                step     = seed_idx + 1,
                metric   = ct_final,
                metrics  = metrics,
                progress = (seed_idx + 1) / n_seeds,
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

    # ---------------------------------------------------------------- #
    # Config schema                                                      #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "num_eval_seeds": {
                "type": "int", "default": 1, "min": 1, "max": 50,
                "label": "Evaluation seeds",
                "help": (
                    "Number of independent random seeds. Use 1 for fixed "
                    "benchmark files; >1 for random-layout mode."
                ),
            },
            "max_iterations": {
                "type": "int", "default": 500, "min": 10, "max": 50_000,
                "label": "Max GRASP iterations",
                "help": (
                    "Paper uses a 3-second time limit; this controls the "
                    "iteration count equivalent. Increase for better quality."
                ),
            },
            "max_no_improve": {
                "type": "int", "default": 10000, "min": 100, "max": 100_000,
                "label": "Max iterations without improvement",
                "help": "Paper stopping criterion (ii): stop after this many non-improving iterations.",
            },
            "alpha_init": {
                "type": "float", "default": 1.0, "min": 0.0, "max": 1.0,
                "label": "Initial greedy rate α₀",
                "help": (
                    "α=1: pure greedy (MNI best only). "
                    "α=0: fully random. Paper starts at α=1 with direction D."
                ),
            },
            "delta_v": {
                "type": "float", "default": 0.05, "min": 0.001, "max": 0.5,
                "label": "Reactive step Δv",
                "help": (
                    "Increment / decrement applied to α at each iteration. "
                    "Paper default: 0.05."
                ),
            },
            "local_search": {
                "type": "bool", "default": True,
                "label": "Enable local search",
                "help": (
                    "If False, skip the reverse-scan local search phase "
                    "(equivalent to paper's RGRASP_S variant)."
                ),
            },
        })
        return base
