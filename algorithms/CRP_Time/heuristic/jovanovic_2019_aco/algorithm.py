"""
JovanovicACO_CRPTime
<2019> <heuristic> <time> <single-bay> <CRP-Time>
Ant Colony System with crane-time objective
n_iterations --- 5000 --- Colony iterations per layout
n_ants --- 10 --- Ants per iteration
q0 --- 0.9 --- Exploitation rate

------------------------------- Reference --------------------------------
R. Jovanović, M. Tuba, S. Voß,
"An efficient ant colony optimization algorithm for the blocks relocation
 problem",
European Journal of Operational Research 274 (2019) 78–90.
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
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.layout_trace import trace_layout
from core.objectives import (
    KinematicsModel,
    ObjectiveSpec,
    evaluate_plan_objectives,
    movement_objective_cost,
)
from core.plan import Movement, RelocationPlan
from .scoring import (
    compute_lb,
    has_time_lb,
    lb_container,
    lb_time,
    run_greedy_rbrp_time,
)


def _solution_to_plan(
    solution: List[Tuple[int, int, int, int]],
    stacks_init: Dict[Any, List[int]],
    all_keys: List[Any],
    n_total: int,
) -> RelocationPlan:
    """Rebuild an explicit tier-annotated plan from the ACO encoding."""
    stacks = {key: list(values) for key, values in stacks_init.items()}
    loc = {
        container: key
        for key, values in stacks.items()
        for container in values
    }
    plan = RelocationPlan()
    by_target: Dict[int, List[Tuple[int, int, int, int]]] = {}
    for record in solution:
        by_target.setdefault(record[3], []).append(record)

    for target in range(1, n_total + 1):
        for c, d_val, _mc, _t in by_target.get(target, []):
            src = loc[c]
            if d_val <= n_total:
                dst = loc[d_val]
            else:
                dst = all_keys[d_val - n_total - 1]
            plan.add(
                Movement(
                    c,
                    src,
                    dst,
                    from_tier=len(stacks[src]),
                    to_tier=len(stacks[dst]) + 1,
                )
            )
            stacks[src].pop()
            stacks[dst].append(c)
            loc[c] = dst

        src = loc[target]
        plan.add(
            Movement(
                target,
                src,
                None,
                from_tier=len(stacks[src]),
            )
        )
        stacks[src].pop()
        loc[target] = None
    return plan


class JovanovicACO_CRPTime(BaseAlgorithm):

    name                = "Jovanović et al. (2019) ACO (CRP-Time)"
    category            = "Heuristic"
    description         = (
        "[single-bay origin]  "
        "Jovanović, Tuba, Voß (EJOR 2019) rBRP-ACO adapted for CRP-Time "
        "(Section 6). Crane-time objective via platform 2-D KinematicsModel "
        "(bay × row, gantry + trolley). Same 4-D pheromone τ[c][d][m_c][t] "
        "and MinMax heuristic as the CRP-R version; val(S) and early "
        "termination use seconds instead of relocation counts."
    )
    compatible_problems = ["CRP-Time"]
    geometry            = "single-bay"
    objectives          = ["crane_time"]
    fidelity            = "adapted"
    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    # ---------------------------------------------------------------- #
    # Training loop                                                      #
    # ---------------------------------------------------------------- #

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        cfg = self.config
        n_seeds = 1  # multi-seed eval removed; single run only
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

            trace_layout(
                f"JovanovicACO_CRPTime.train: seed {seed+1}/{n_seeds}  "
                f"n_iters={n_iters}  n_ants={n_ants}  q0={q0}"
            )

            np.random.seed(cfg.seed + seed * 1000)

            # ── Set up environment ────────────────────────────────────── #
            env = problem_factory()
            env.reset(options={"skip_auto_retrieve": True})
            initial_yard = copy.deepcopy(env.yard)
            objective_spec = ObjectiveSpec.from_config(env.config)
            kinematics = KinematicsModel.from_config_extra(env.config.extra)

            n_total   = int(env.config.num_containers)
            max_tiers = int(env.config.max_tiers)
            n_stacks  = env.config.num_bays * env.config.num_rows

            # Read kinematics from problem config (same keys as CRP-Time schema)
            extra      = env.config.extra
            gantry_s   = float(extra.get("gantry_s_per_bay",  3.5))
            trolley_s  = float(extra.get("trolley_s_per_row", 1.2))
            accel_s    = float(extra.get("gantry_accel_s",   40.0))
            spreader_s = float(extra.get("spreader_s",        30.0))

            all_keys:          List[Any] = []
            key_to_1based_idx: Dict[Any, int] = {}
            stacks_init:       Dict[Any, List[int]] = {}

            for a in range(n_stacks):
                key = env._action_to_stack(a)
                all_keys.append(key)
                key_to_1based_idx[key] = a + 1
                stk = env.yard.stacks.get(key)
                stacks_init[key] = (
                    [int(c.priority) for c in stk.containers] if stk else []
                )

            # ── Greedy warm-start ────────────────────────────────────── #
            S_best, _legacy_greedy_cost = run_greedy_rbrp_time(
                stacks_init, all_keys, n_total, max_tiers, key_to_1based_idx,
                gantry_s, trolley_s, accel_s, spreader_s,
            )

            # ── Initial lower bound ──────────────────────────────────── #
            lb_init_nwl  = compute_lb(stacks_init)          # integer NWL count
            # LB_f / LB_v of the initial bay for val (eqs. 44–47)
            lb_init_time = lb_time(stacks_init, key_to_1based_idx, objective_spec)

            # ── Opt-1: precompute location + stack-min arrays ─────────── #
            loc_init:       List[Any]      = [None] * (n_total + 1)
            stack_min_init: Dict[Any, int] = {}
            empty_d:        Dict[Any, int] = {}
            nw_init:        List[bool]     = [False] * (n_total + 1)

            for key, prios in stacks_init.items():
                stack_min_init[key] = min(prios) if prios else (n_total + 1)
                empty_d[key]        = n_total + key_to_1based_idx[key]
                for i, p in enumerate(prios):
                    loc_init[p] = key
                    nw_init[p]  = i > 0 and min(prios[:i]) < p

            # LB(Bay) of the current bay for the early abort (Alg. 2)
            use_lb = has_time_lb(objective_spec)

            best_plan = _solution_to_plan(
                S_best, stacks_init, all_keys, n_total
            )
            best_metrics = evaluate_plan_objectives(
                best_plan,
                objective_spec,
                kinematics=kinematics,
                initial_yard=initial_yard,
            )
            best_cost = float(best_metrics["objective_value"])

            # ── Pheromone initialisation (Eq. 37) ────────────────────── #
            val_greedy = 1.0 / max(best_cost - lb_init_time + 1.0, 1.0)
            tau_0      = val_greedy / n_stacks
            tau_min    = tau_0 / n_stacks
            pheromone = np.full(
                (n_total, n_total + n_stacks, max_moves + 1, n_total),
                tau_0, dtype=np.float32,
            )

            iters_no_improve = 0

            # ── Main ACO loop ─────────────────────────────────────────── #
            for iteration in range(n_iters):
                if stop_event.is_set():
                    break

                iter_improved = False

                # ── Colony: each ant builds one solution ──────────────── #
                for _ant in range(n_ants):
                    if stop_event.is_set():
                        break

                    # Reset per-ant state
                    stacks       = {k: list(v) for k, v in stacks_init.items()}
                    M            = [0] * (n_total + 1)
                    S: List[Tuple[int, int, int, int]] = []
                    valid        = True
                    loc          = list(loc_init)
                    smin         = dict(stack_min_init)
                    crane_pos    = (1, 1)            # crane starts at (bay=1, row=1)
                    time_so_far  = 0.0               # accumulated selected objective
                    nw           = list(nw_init)
                    lb_cur       = lb_init_time      # LB(Bay) of the current bay

                    for target in range(1, n_total + 1):
                        if not valid:
                            break

                        src_key = loc[target]
                        if src_key is None:
                            continue

                        # ── Relocate all blockers above target ────────── #
                        while (
                            valid
                            and stacks[src_key]
                            and stacks[src_key][-1] != target
                        ):
                            c = stacks[src_key][-1]

                            # Build candidate list
                            best_g   = -1.0
                            best_dst = None
                            best_d   = 0
                            c_i      = c - 1
                            mc_cl    = min(M[c], max_moves)
                            t_i      = target - 1
                            cands_g: List[float] = []
                            cands_k: List[Any]   = []
                            cands_d: List[int]   = []

                            for dst_key in all_keys:
                                if dst_key == src_key:
                                    continue
                                if len(stacks[dst_key]) >= max_tiers:
                                    continue

                                # O(1) dd* via smin
                                sm    = smin[dst_key]
                                d_val = sm if sm <= n_total else empty_d[dst_key]

                                # Eq. 27
                                dif_val = (sm - c) if sm > c else (2 * n_total + 1 - sm)
                                f_val   = 1.0 / (1.0 + dif_val)

                                tau = float(pheromone[c_i, d_val - 1, mc_cl, t_i])
                                g   = f_val * tau
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

                            # Transition rule (Eq. 31–32)
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

                            # Record 4-tuple and apply relocation
                            mc_rec = mc_cl
                            S.append((c, d_val, mc_rec, target))
                            movement = Movement(
                                c,
                                src_key,
                                dst_key,
                                from_tier=len(stacks[src_key]),
                                to_tier=len(stacks[dst_key]) + 1,
                            )
                            nw_new = smin[dst_key] < c

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

                            loc[c]  = dst_key
                            M[c]   += 1

                            # Accumulate the user-selected additive objective.
                            cost, _selected_time, crane_pos = movement_objective_cost(
                                movement,
                                objective_spec,
                                kinematics,
                                crane_pos,
                            )
                            time_so_far += cost

                            if use_lb:
                                lb_cur += lb_container(
                                    key_to_1based_idx[dst_key], movement.to_tier,
                                    nw_new, objective_spec,
                                ) - lb_container(
                                    key_to_1based_idx[src_key], movement.from_tier,
                                    nw[c], objective_spec,
                                )
                                nw[c] = nw_new

                            # Alg. 2: stop once the partial cost plus LB(Bay)
                            # reaches the best cost.
                            if time_so_far + lb_cur >= best_cost:
                                valid = False

                        # Retrieve target (also incurs crane time)
                        if (
                            valid
                            and stacks[src_key]
                            and stacks[src_key][-1] == target
                        ):
                            movement = Movement(
                                target,
                                src_key,
                                None,
                                from_tier=len(stacks[src_key]),
                            )
                            cost, _selected_time, crane_pos = movement_objective_cost(
                                movement,
                                objective_spec,
                                kinematics,
                                crane_pos,
                            )
                            time_so_far += cost
                            # The retrieval cost equals the term it removes
                            # from LB(Bay), so no abort check is needed here.
                            if use_lb:
                                lb_cur -= lb_container(
                                    key_to_1based_idx[src_key], movement.from_tier,
                                    nw[target], objective_spec,
                                )

                            stacks[src_key].pop()
                            if smin[src_key] == target:
                                smin[src_key] = (
                                    min(stacks[src_key])
                                    if stacks[src_key]
                                    else (n_total + 1)
                                )
                            loc[target] = None

                    # ── Local pheromone update (Eq. 36) ──────────────── #
                    for c, d_val, mc_rec, t in S:
                        c_i = c - 1
                        d_i = d_val - 1
                        t_i = t - 1
                        if (
                            0 <= c_i < n_total
                            and 0 <= d_i < n_total + n_stacks
                            and 0 <= t_i < n_total
                        ):
                            pheromone[c_i, d_i, mc_rec, t_i] = max(
                                np.float32(tau_min),
                                np.float32(pheromone[c_i, d_i, mc_rec, t_i] * phi),
                            )

                    # ── Update best selected objective ─────────────────── #
                    if valid and time_so_far < best_cost:
                        best_cost        = time_so_far
                        S_best           = S.copy()
                        iter_improved    = True
                        iters_no_improve = 0

                # ── Stagnation reinitialisation ───────────────────────── #
                if not iter_improved:
                    iters_no_improve += 1
                if iters_no_improve >= max_const:
                    val_b   = 1.0 / max(best_cost - lb_init_time + 1.0, 1.0)
                    tau_new = val_b / n_stacks
                    tau_min = tau_new / n_stacks
                    pheromone[:] = np.float32(tau_new)
                    iters_no_improve = 0

                # ── Global pheromone update (Eqs. 34–35) ─────────────── #
                val_b   = 1.0 / max(best_cost - lb_init_time + 1.0, 1.0)
                delta   = np.float32(val_b)
                tau_min = val_b / (n_stacks ** 2)
                for c, d_val, mc_rec, t in S_best:
                    c_i = c - 1
                    d_i = d_val - 1
                    t_i = t - 1
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

                # ── Periodic GUI push ─────────────────────────────────── #
                if (iteration + 1) % report_every == 0:
                    frac = (seed * n_iters + iteration + 1) / (n_seeds * n_iters)
                    report_plan = _solution_to_plan(
                        S_best, stacks_init, all_keys, n_total
                    )
                    report_metrics = evaluate_plan_objectives(
                        report_plan,
                        objective_spec,
                        kinematics=kinematics,
                        initial_yard=initial_yard,
                    )
                    report_metrics.update({
                        "lower_bound": float(lb_init_time),
                        "iteration": float(iteration + 1),
                    })
                    self._push(
                        result_queue,
                        step     = seed + 1,
                        metric   = float(best_cost),
                        metrics  = report_metrics,
                        progress = min(frac, (seed + 1) / n_seeds),
                    )

            # ── End of seed ───────────────────────────────────────────── #
            best_plan = _solution_to_plan(
                S_best, stacks_init, all_keys, n_total
            )
            metrics = evaluate_plan_objectives(
                best_plan,
                objective_spec,
                kinematics=kinematics,
                initial_yard=initial_yard,
            )
            metrics["steps"] = float(best_plan.num_moves())
            metrics["progress"] = 1.0
            all_metrics.append(metrics)

            if float(best_cost) < self._best_metric:
                self._best_metric   = float(best_cost)
                self._best_solution = [
                    (move.to_pos[0] - 1) * env.config.num_rows
                    + (move.to_pos[1] - 1)
                    for move in best_plan.movements
                    if not move.is_retrieval
                ]

            self._push(
                result_queue,
                step     = seed + 1,
                metric   = float(best_cost),
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
    # Configuration schema                                               #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "n_iterations": {
                "type": "int", "default": 5000, "min": 100, "max": 50000,
                "label": "ACO iterations",
                "help": (
                    "Colony iterations per layout. Paper uses 5 000 for rBRP; "
                    "crane-time convergence is slower — consider 10 000+ for "
                    "best quality (see paper Section 7.4)."
                ),
            },
            "n_ants": {
                "type": "int", "default": 10, "min": 1, "max": 100,
                "label": "Ants per iteration",
                "help": "Colony size — number of ants per iteration. Paper default: 10.",
            },
            "q0": {
                "type": "float", "default": 0.9, "min": 0.0, "max": 1.0,
                "label": "Exploitation rate q₀",
                "help": (
                    "Probability of greedy exploitation vs. probabilistic "
                    "exploration. Paper default: 0.9."
                ),
            },
            "rho": {
                "type": "float", "default": 0.1, "min": 0.0, "max": 1.0,
                "label": "Global update rate ρ",
                "help": "Pheromone deposit rate for best solution. Paper default: 0.1.",
            },
            "phi": {
                "type": "float", "default": 0.9, "min": 0.0, "max": 1.0,
                "label": "Local evaporation φ",
                "help": "Pheromone decay after each ant solution. Paper default: 0.9.",
            },
            "max_moves": {
                "type": "int", "default": 10, "min": 1, "max": 30,
                "label": "Max relocations tracked (MaxMoves)",
                "help": "Pheromone matrix depth for relocation-count dimension. Paper default: 10.",
            },
            "max_const_iter": {
                "type": "int", "default": 100, "min": 10, "max": 1000,
                "label": "Stagnation threshold",
                "help": (
                    "Reinitialise pheromone after this many iterations without "
                    "improvement. Paper default: 100."
                ),
            },
        })
        return base
