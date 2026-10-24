"""
JovanovicACO_CRPTime
<2019> <heuristic> <time> <single-bay> <CRP-Time>
Ant Colony System with crane-time objective
n_iterations --- 5000 --- Colony iterations per layout
n_ants --- 10 --- Ants per iteration
q0 --- 0.9 --- Exploitation rate
rho --- 0.1 --- Global update rate
phi --- 0.9 --- Local evaporation
max_moves --- 10 --- MaxMoves (pheromone depth)
max_const_iter --- 100 --- Stagnation threshold (MaxConst)

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
from typing import Callable, Dict, List, Optional

import numpy as np

from algorithms.CRP_Time.utils.bay import read_bay
from algorithms.CRP_Time.utils.crane_time import check_objective
from algorithms.CRP_Time.utils.finish import finish
from algorithms.CRP_Time.utils.params import parameters
from algorithms.CRP_Time.utils.plan import actions, replay
from algorithms.CRP_Time.utils.timer import Stopwatch
from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.layout_trace import trace_layout
from core.objectives import KinematicsModel, ObjectiveSpec, evaluate_plan_objectives

from .ant import build_solution, decode, start
from .greedy import greedy
from .pheromone import global_update, initial, local_update, reinitialise


class JovanovicACO_CRPTime(BaseAlgorithm):

    name                = "Jovanović et al. (2019) ACO (CRP-Time)"
    category            = "Heuristic"
    description         = (
        "Ant colony optimisation: ants build relocation plans guided by "
        "pheromones and a greedy heuristic, and the search stops after a fixed "
        "number of iterations, with the seed fixed at 0 so that results are "
        "reproducible. Deviation from the paper: under f2, the lower bound on "
        "the remaining crane time, used to cut off ants early and to rate "
        "solutions for the pheromone update, omits the paper’s extra travel "
        "term per misplaced container, because the paper’s bound is not "
        "always a valid lower bound."
    )
    compatible_problems = ["CRP-Time"]
    geometry            = "single-bay"
    objectives          = ["crane_time"]
    fidelity            = "faithful"
    relocation_rule     = "restricted"
    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        cfg = self.config
        n_seeds = 1  # multi-seed eval removed; single run only
        par          = parameters(cfg, type(self))
        n_iters      = int(par["n_iterations"])
        n_ants       = int(par["n_ants"])
        q0           = float(par["q0"])
        rho          = float(par["rho"])
        phi          = float(par["phi"])
        max_moves    = int(par["max_moves"])
        max_const    = int(par["max_const_iter"])
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
            stacks_init, positions = read_bay(env)
            initial_yard = copy.deepcopy(env.yard)
            objective_spec = ObjectiveSpec.from_config(env.config)
            check_objective(objective_spec, self.name, ("crane_time", "relocations"))
            # Only for the legacy RMGC metric of the shared evaluator.
            kinematics = KinematicsModel.from_config_extra(env.config.extra)

            n_total  = int(env.config.num_containers)
            n_stacks = len(stacks_init)
            timer    = Stopwatch()
            timer.start()
            ant0     = start(stacks_init, positions, int(env.config.max_tiers), objective_spec)

            def evaluate(solution):
                plan = replay(stacks_init, positions, decode(solution, stacks_init, n_total))
                return plan, evaluate_plan_objectives(
                    plan, objective_spec, kinematics=kinematics, initial_yard=initial_yard,
                )

            # ── Greedy warm-start and pheromone initialisation ─────── #
            S_best = greedy(stacks_init, n_total, int(env.config.max_tiers), max_moves)
            best_cost = float(evaluate(S_best)[1]["objective_value"])
            pheromone, tau_min = initial(best_cost, ant0.lb, n_total, n_stacks, max_moves)

            iters_no_improve = 0

            # ── Main ACO loop ─────────────────────────────────────────── #
            for iteration in range(n_iters):
                if stop_event.is_set():
                    break

                iter_improved = False

                for _ant in range(n_ants):
                    if stop_event.is_set():
                        break
                    S, valid, cost = build_solution(ant0, pheromone, q0, max_moves, best_cost, objective_spec)
                    local_update(pheromone, S, phi, tau_min)
                    if valid and cost < best_cost:
                        best_cost        = cost
                        S_best           = S
                        iter_improved    = True
                        iters_no_improve = 0

                if not iter_improved:
                    iters_no_improve += 1
                if iters_no_improve >= max_const:
                    reinitialise(pheromone, best_cost, ant0.lb, n_stacks)
                    iters_no_improve = 0

                tau_min = global_update(pheromone, S_best, best_cost, ant0.lb, rho, n_stacks)

                # ── Periodic GUI push ─────────────────────────────────── #
                if (iteration + 1) % report_every == 0:
                    frac = (seed * n_iters + iteration + 1) / (n_seeds * n_iters)
                    with timer.paused():
                        self._push(
                            result_queue,
                            step     = seed + 1,
                            metric   = float(best_cost),
                            metrics  = evaluate(S_best)[1],
                            progress = min(frac, (seed + 1) / n_seeds),
                        )

            # ── End of seed ───────────────────────────────────────────── #
            timer.stop()
            best_plan = replay(stacks_init, positions, decode(S_best, stacks_init, n_total))
            metrics, env_final = finish(problem_factory, best_plan, objective_spec, self.name, timer.elapsed)
            all_metrics.append(metrics)

            # <=: the progress pushes have already lowered _best_metric to best_cost
            if float(best_cost) <= self._best_metric:
                self._best_metric   = float(best_cost)
                self._best_solution = actions(best_plan, env.config.num_rows)

            self._push(
                result_queue,
                step     = seed + 1,
                metric   = float(best_cost),
                metrics  = metrics,
                progress = (seed + 1) / n_seeds,
                snapshot = env_final.get_state_snapshot(),
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
        # max_iterations and report_interval are not read; the seed is not a field, so it stays 0.
        for key in ("max_iterations", "report_interval", "seed"):
            base.pop(key)
        base.update({
            "n_iterations": {
                "type": "int", "default": 5000, "min": 100, "max": 50000,
                "label": "ACO iterations",
                "help": (
                    "Colony iterations per layout. 5000 in the paper "
                    "(relocations); Sec. 7.4 reports about 5x more iterations "
                    "for crane time."
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
