"""
Jovanović, Tuba, Voß (2019) — ACO for CRP-Time (crane working time objective).

Reference
---------
R. Jovanović, M. Tuba, S. Voß,
"An efficient ant colony optimization algorithm for the blocks relocation
 problem",
European Journal of Operational Research 274 (2019) 78–90.
https://doi.org/10.1016/j.ejor.2018.09.038

CRP-Time adaptation
--------------------
The paper shows the ACO "can easily be adapted for solving the BRP in which
the objective function is related to the crane operation time" (abstract).

Construction: identical to the CRP-R ACO — 4-D pheromone τ[c][d][m_c][t],
ACS transition rule, MinMax greedy warm-start.  Each relocation also stores
the physical destination key so the move sequence can be replayed.

Evaluation: after every ant finishes, the move list is replayed through a
lightweight RelocationPlan and `compute_crane_time` is called.  The best
solution is selected and pheromone is updated based on crane_time quality.

Crane-time model (Lee & Lee 2010, same as core/objectives.py)
-------------------------------------------------------------
  travel(A→B) = max(gantry_t, trolley_t)
  gantry_t    = |bay_B − bay_A| × t_bay + t_acc   (if bay changes; else 0)
  trolley_t   = |row_B − row_A| × t_row
  move_time   = travel(crane→src) + travel(src→dst) + spreader_s

For single-bay yards (num_bays=1) gantry_t is always 0, recovering the
"no cross-bay" formula discussed in the paper.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.objectives import KinematicsModel, compute_crane_time
from core.plan import Movement, RelocationPlan

from algorithms.CRP_R.heuristic.jovanovic_2019_aco.scoring import (
    compute_lb,
)


# ================================================================ #
#  Crane-time replay helper                                         #
# ================================================================ #

def _replay_crane_time(
    moves:       List[Tuple[int, Any]],      # [(container_priority, dst_key), …]
    stacks_init: Dict[Any, List[int]],
    n_total:     int,
    kin:         KinematicsModel,
) -> float:
    """
    Simulate the ordered move list on a local yard copy, build a
    RelocationPlan, and return the exact crane working time.
    """
    stacks: Dict[Any, List[int]] = {k: list(v) for k, v in stacks_init.items()}
    loc:    Dict[int, Any]       = {}
    for key, prios in stacks_init.items():
        for p in prios:
            loc[p] = key

    plan      = RelocationPlan()
    move_ptr  = 0
    cur_target = 1

    while cur_target <= n_total:
        src_key = loc.get(cur_target)
        if src_key is None:
            cur_target += 1
            continue

        # Relocate all blockers above cur_target
        while stacks[src_key] and stacks[src_key][-1] != cur_target:
            if move_ptr >= len(moves):
                break
            c, dst_key = moves[move_ptr]
            move_ptr  += 1
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

    return float(compute_crane_time(plan, kin))


# ================================================================ #
#  Greedy warm-start (returns both full tuples and move list)       #
# ================================================================ #

def _greedy_solution(
    stacks_init:       Dict[Any, List[int]],
    all_keys:          List[Any],
    n_total:           int,
    max_tiers:         int,
    key_to_1based_idx: Dict[Any, int],
) -> Tuple[List[Tuple], List[Tuple[int, Any]], int]:
    """
    MinMax greedy rBRP.

    Returns
    -------
    full_tuples : [(c, d_val, mc_rec, target, dst_key), …]
    move_list   : [(c, dst_key), …]   for crane_time replay
    cost        : number of relocations
    """
    stacks: Dict[Any, List[int]] = {k: list(v) for k, v in stacks_init.items()}
    M:      Dict[int, int]       = {}
    full:   List[Tuple]          = []
    moves:  List[Tuple[int, Any]]= []

    for target in range(1, n_total + 1):
        src_key = next((k for k, p in stacks.items() if target in p), None)
        if src_key is None:
            continue

        while stacks[src_key] and stacks[src_key][-1] != target:
            c = stacks[src_key][-1]
            best_dst:   Optional[Any] = None
            best_score: float         = float("inf")
            best_d:     int           = 0

            for dst_key in all_keys:
                if dst_key == src_key or len(stacks[dst_key]) >= max_tiers:
                    continue
                sm    = min(stacks[dst_key]) if stacks[dst_key] else (n_total + 1)
                d_val = sm if sm <= n_total else n_total + key_to_1based_idx[dst_key]
                dif_v = (d_val - c) if d_val > c else (2 * n_total + 1 - d_val)
                if dif_v < best_score:
                    best_score = dif_v
                    best_dst   = dst_key
                    best_d     = d_val

            if best_dst is None:
                break

            mc = M.get(c, 0)
            full.append((c, best_d, mc, target, best_dst))
            moves.append((c, best_dst))
            stacks[src_key].pop()
            stacks[best_dst].append(c)
            M[c] = mc + 1

        if stacks[src_key] and stacks[src_key][-1] == target:
            stacks[src_key].pop()

    return full, moves, len(full)


# ================================================================ #
#  Main algorithm                                                   #
# ================================================================ #

class JovanovicACO_CRPTime(BaseAlgorithm):

    name                = "Jovanović et al. (2019) ACO (CRP-Time)"
    category            = "Heuristic"
    description         = (
        "[native multi-bay]  "
        "Jovanović, Tuba, Voß (EJOR 2019) ACO adapted for CRP-Time. "
        "4-D pheromone τ[c][d][m_c][t], ACS transition, MinMax greedy "
        "warm-start.  Primary metric: crane working time (Lee & Lee "
        "kinematics with full cross-bay gantry model).  Pheromone quality "
        "based on crane_time of the best solution found."
    )
    compatible_problems = ["CRP-Time"]
    step_label          = "Seed"

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
        n_iters      = int(cfg.extra.get("n_iterations",   5000))
        n_ants       = int(cfg.extra.get("n_ants",         10))
        q0           = float(cfg.extra.get("q0",            0.9))
        rho          = float(cfg.extra.get("rho",           0.1))
        phi          = float(cfg.extra.get("phi",           0.9))
        max_moves    = int(cfg.extra.get("max_moves",       10))
        max_const    = int(cfg.extra.get("max_const_iter",  100))
        report_every = max(1, n_iters // 50)

        all_metrics: List[Dict] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            np.random.seed(cfg.seed + seed * 1000)

            env = problem_factory()
            env.config.seed = seed
            env.reset(options={"skip_auto_retrieve": True})

            n_total   = int(env.config.num_containers)
            max_tiers = int(env.config.max_tiers)
            n_stacks  = env.config.num_bays * env.config.num_rows
            kin       = KinematicsModel.from_config_extra(env.config.extra)

            all_keys:          List[Any]     = []
            key_to_1based_idx: Dict[Any, int]= {}
            stacks_init:       Dict[Any, List[int]] = {}

            for a in range(n_stacks):
                key = env._action_to_stack(a)
                all_keys.append(key)
                key_to_1based_idx[key] = a + 1
                stk = env.yard.stacks.get(key)
                stacks_init[key] = (
                    [int(c.priority) for c in stk.containers] if stk else []
                )

            # ── Greedy warm-start ─────────────────────────────────── #
            S_best_full, S_best_moves, best_reloc = _greedy_solution(
                stacks_init, all_keys, n_total, max_tiers, key_to_1based_idx
            )
            lb_init  = compute_lb(stacks_init)
            best_ct  = _replay_crane_time(
                S_best_moves, stacks_init, n_total, kin
            )

            # ── Pheromone initialisation ──────────────────────────── #
            val_greedy = 1.0 / max(best_reloc - lb_init + 1, 1)
            tau_0      = val_greedy / n_stacks
            tau_min    = tau_0 / n_stacks

            pheromone = np.full(
                (n_total, n_total + n_stacks, max_moves + 1, n_total),
                tau_0, dtype=np.float32,
            )

            # ── Opt-1: location and stack-min arrays ──────────────── #
            loc_init:       List[Any]       = [None] * (n_total + 1)
            stack_min_init: Dict[Any, int]  = {}
            empty_d:        Dict[Any, int]  = {}

            for key, prios in stacks_init.items():
                stack_min_init[key] = min(prios) if prios else (n_total + 1)
                empty_d[key]        = n_total + key_to_1based_idx[key]
                for p in prios:
                    loc_init[p] = key

            iters_no_improve = 0

            # ── Main ACO loop ─────────────────────────────────────── #
            for iteration in range(n_iters):
                if stop_event.is_set():
                    break

                iter_improved = False

                for _ant in range(n_ants):
                    if stop_event.is_set():
                        break

                    stacks  = {k: list(v) for k, v in stacks_init.items()}
                    M       = [0] * (n_total + 1)
                    # Full tuple: (c, d_val, mc_rec, target, dst_key)
                    S:       List[Tuple] = []
                    valid   = True
                    loc     = list(loc_init)
                    smin    = dict(stack_min_init)
                    lb_curr = lb_init

                    for target in range(1, n_total + 1):
                        if not valid:
                            break

                        src_key = loc[target]
                        if src_key is None:
                            continue

                        while (
                            valid
                            and stacks[src_key]
                            and stacks[src_key][-1] != target
                        ):
                            c     = stacks[src_key][-1]
                            c_i   = c - 1
                            mc_cl = min(M[c], max_moves)
                            t_i   = target - 1

                            best_g    = -1.0
                            best_dst  = None
                            best_d    = 0
                            cands_g:  List[float] = []
                            cands_k:  List[Any]   = []
                            cands_d:  List[int]   = []

                            for dst_key in all_keys:
                                if dst_key == src_key:
                                    continue
                                if len(stacks[dst_key]) >= max_tiers:
                                    continue

                                sm    = smin[dst_key]
                                d_val = sm if sm <= n_total else empty_d[dst_key]
                                dif_v = (d_val - c) if d_val > c else (2 * n_total + 1 - d_val)
                                f_val = 1.0 / (1.0 + dif_v)
                                tau   = float(pheromone[c_i, d_val - 1, mc_cl, t_i])
                                g     = f_val * tau

                                cands_g.append(g)
                                cands_k.append(dst_key)
                                cands_d.append(d_val)

                                if g > best_g:
                                    best_g   = g
                                    best_dst = dst_key
                                    best_d   = d_val

                            if not cands_k:
                                valid = False
                                break

                            # ACS transition rule
                            if np.random.uniform() < q0 or len(cands_k) == 1:
                                dst_key = best_dst
                                d_val   = best_d
                            else:
                                g_arr = np.asarray(cands_g, dtype=np.float64)
                                g_sum = g_arr.sum()
                                probs = (
                                    g_arr / g_sum
                                    if g_sum > 0.0
                                    else np.ones(len(cands_k)) / len(cands_k)
                                )
                                idx     = np.random.choice(len(cands_k), p=probs)
                                dst_key = cands_k[idx]
                                d_val   = cands_d[idx]

                            # Incremental LB update
                            was_non_wl  = smin[src_key] < c
                            will_non_wl = smin[dst_key] < c

                            mc_rec = mc_cl
                            S.append((c, d_val, mc_rec, target, dst_key))

                            stacks[src_key].pop()
                            if smin[src_key] == c:
                                smin[src_key] = (
                                    min(stacks[src_key])
                                    if stacks[src_key]
                                    else (n_total + 1)
                                )
                            stacks[dst_key].append(c)
                            if c < smin[dst_key]:
                                smin[dst_key] = c

                            loc[c] = dst_key
                            M[c]  += 1

                            if was_non_wl:
                                lb_curr -= 1
                            if will_non_wl:
                                lb_curr += 1

                            if len(S) + lb_curr >= best_reloc:
                                valid = False

                        if (
                            valid
                            and stacks[src_key]
                            and stacks[src_key][-1] == target
                        ):
                            stacks[src_key].pop()
                            if smin[src_key] == target:
                                smin[src_key] = (
                                    min(stacks[src_key])
                                    if stacks[src_key]
                                    else (n_total + 1)
                                )
                            loc[target] = None

                    # ── Local pheromone update ────────────────────── #
                    for c, d_val, mc_rec, t, _ in S:
                        c_i = c - 1; d_i = d_val - 1; t_i = t - 1
                        if (
                            0 <= c_i < n_total
                            and 0 <= d_i < n_total + n_stacks
                            and 0 <= t_i < n_total
                        ):
                            pheromone[c_i, d_i, mc_rec, t_i] = max(
                                np.float32(tau_min),
                                np.float32(
                                    pheromone[c_i, d_i, mc_rec, t_i] * phi
                                ),
                            )

                    # ── Evaluate crane_time and update best ───────── #
                    if valid:
                        ant_moves = [(c, dst_key) for c, _, _, _, dst_key in S]
                        ant_ct    = _replay_crane_time(
                            ant_moves, stacks_init, n_total, kin
                        )

                        if ant_ct < best_ct:
                            best_ct       = ant_ct
                            best_reloc    = len(S)
                            S_best_full   = S[:]
                            S_best_moves  = ant_moves
                            iter_improved = True
                            iters_no_improve = 0

                # ── Stagnation reinitialisation ───────────────────── #
                if not iter_improved:
                    iters_no_improve += 1
                if iters_no_improve >= max_const:
                    val_b   = 1.0 / max(best_reloc - lb_init + 1, 1)
                    tau_new = val_b / n_stacks
                    tau_min = tau_new / n_stacks
                    pheromone[:] = np.float32(tau_new)
                    iters_no_improve = 0

                # ── Global pheromone update (based on best solution) ─ #
                val_b   = 1.0 / max(best_reloc - lb_init + 1, 1)
                delta   = np.float32(val_b)
                tau_min = max(val_b / (n_stacks ** 2), 1e-10)
                for c, d_val, mc_rec, t, _ in S_best_full:
                    c_i = c - 1; d_i = d_val - 1; t_i = t - 1
                    if (
                        0 <= c_i < n_total
                        and 0 <= d_i < n_total + n_stacks
                        and 0 <= t_i < n_total
                    ):
                        old = pheromone[c_i, d_i, mc_rec, t_i]
                        pheromone[c_i, d_i, mc_rec, t_i] = max(
                            np.float32(tau_min),
                            np.float32((1.0 - rho) * old + rho * delta),
                        )

                # ── Periodic GUI push ─────────────────────────────── #
                if (iteration + 1) % report_every == 0:
                    frac = (seed * n_iters + iteration + 1) / (n_seeds * n_iters)
                    self._push(
                        result_queue,
                        step     = seed + 1,
                        metric   = float(best_ct),
                        metrics  = {
                            "crane_time":  float(best_ct),
                            "relocations": float(best_reloc),
                            "lower_bound": float(lb_init),
                            "iteration":   float(iteration + 1),
                            "time":        float(best_ct),
                        },
                        progress = min(frac, (seed + 1) / n_seeds),
                    )

            # ── Seed done ─────────────────────────────────────────── #
            metrics = {
                "crane_time":  float(best_ct),
                "relocations": float(best_reloc),
                "lower_bound": float(lb_init),
                "time":        float(best_ct),
                "steps":       float(best_reloc),
                "progress":    1.0,
            }
            all_metrics.append(metrics)

            if best_ct < self._best_metric:
                self._best_metric   = best_ct
                self._best_solution = []

            self._push(
                result_queue,
                step     = seed + 1,
                metric   = float(best_ct),
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
                    "Independent ACO runs per layout. Use 1 for fixed benchmark "
                    "files; >1 for random-layout mode."
                ),
            },
            "n_iterations": {
                "type": "int", "default": 5000, "min": 100, "max": 50000,
                "label": "ACO iterations",
                "help": "Colony iterations per layout (paper default: 5 000).",
            },
            "n_ants": {
                "type": "int", "default": 10, "min": 1, "max": 100,
                "label": "Ants per iteration",
                "help": "Number of ants constructing solutions per iteration.",
            },
            "q0": {
                "type": "float", "default": 0.9, "min": 0.0, "max": 1.0,
                "label": "Exploitation rate q₀",
                "help": "Probability of greedy exploitation. Paper default: 0.9.",
            },
            "rho": {
                "type": "float", "default": 0.1, "min": 0.0, "max": 1.0,
                "label": "Global update rate ρ",
                "help": "Pheromone deposit rate for the best solution. Paper default: 0.1.",
            },
            "phi": {
                "type": "float", "default": 0.9, "min": 0.0, "max": 1.0,
                "label": "Local evaporation φ",
                "help": "Per-ant pheromone decay factor. Paper default: 0.9.",
            },
            "max_moves": {
                "type": "int", "default": 10, "min": 1, "max": 30,
                "label": "Max relocations tracked (MaxMoves)",
                "help": "Pheromone matrix depth for relocation count. Paper default: 10.",
            },
            "max_const_iter": {
                "type": "int", "default": 100, "min": 10, "max": 1000,
                "label": "Stagnation threshold",
                "help": (
                    "Reinitialise pheromone after this many non-improving "
                    "iterations. Paper default: 100."
                ),
            },
        })
        return base
