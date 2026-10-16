"""
LopezPlata2019Heuristic
<2019> <heuristic> <time> <single-bay> <CRP-Time>
A*-based heuristic for the restricted BRP, converted to crane time (f2 / f2vert)
theta       --- 40 ---       Percentage in eq. 29 for the depth beta of step 1
lam         --- 500 ---      Nodes per subset N_g and N_q (lambda)
node_budget --- 10000000 --- Nodes visited by step 1 (not in the paper, L18)
rho         --- 300 ---      Iterations of step 2
kappa       --- 2 ---        Retrievals per Expansion
mu          --- 5 ---        Sequences in the roulette wheel
tau         --- 3 ---        Branching factor of Expansion
time_limit_sec --- 0 ---     Time limit per instance checked in step 2, 0 = none (not in the paper, L32)

------------------------------- Reference --------------------------------
I. López-Plata, C. Expósito-Izquierdo, E. Lalla-Ruiz, B. Melián-Batista,
J.M. Moreno-Vega,
"Minimizing the operating cost of block retrieval operations in stacking
 facilities",
Computers & Industrial Engineering 136 (2019) 436–452.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import multiprocessing as mp
import random
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from algorithms.CRP_Time.utils.bay import read_bay
from algorithms.CRP_Time.utils.crane_time import check_objective
from algorithms.CRP_Time.utils.finish import finish
from algorithms.CRP_Time.utils.params import parameters
from algorithms.CRP_Time.utils.plan import actions, replay
from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.objectives import ObjectiveSpec

from .bay import Bay, f
from .greedy import beta, greedy
from .operating_cost import CostModel
from .step1 import partial_solutions
from .step2 import algorithm2_loop
from .step3 import postprocess
from .time_model import TimeModel


class LopezPlata2019Heuristic(BaseAlgorithm):
    """
    The parameters in the UI are those of config_schema(). Two settings are not in the UI:
    cost_model, "crane_time" (default: f2 or f2vert, the CRP-Time time model) or "operating_cost"
    (the paper's own cost, for the validation in the paper's setting), and under "operating_cost"
    the weights eta1, gamma1, eta2, gamma2 of CostModel.
    """

    name = "López-Plata et al. (2019) A∗-Based Heuristic"
    category = "Heuristic"
    description = (
        "A*-based heuristic for the restricted Blocks Relocation Problem "
        "(López-Plata et al., C&IE 2019), converted from the paper's operating cost to crane "
        "time: it minimises the selected time model, f2 or f2vert (rmgc_current is refused); "
        "relocations are only measured. Step 1: Algorithm 1 (A*) to depth beta, a share of the "
        "length of a greedy solution, gives the partial solutions N_g and N_q (lambda each). "
        "Step 2: rho iterations complete every partial solution with Expansion (kappa retrievals, "
        "tau branches, roulette over the mu best sequences). Step 3: repair of temporary movements "
        "(Sec. 6.3). Deviations: the lower bound h is an admissible crane-time bound that, like the "
        "paper's estimate, counts one retrieval per block and one relocation per badly placed "
        "block (L27); step 1 is bounded by a node budget with iterative deepening (L18; beta_eff "
        "and budget_hit are reported per run); the factor that scales beta with the number of "
        "blocks is used with its end values swapped as a presumed misprint (L19); an optional "
        "time limit stops step 2 after the partial solution being completed (L32, off by default)."
    )
    compatible_problems = ["CRP-Time"]
    geometry = "single-bay"
    objectives = ["crane_time"]
    fidelity = "faithful"
    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    # ---------------------------------------------------------------- #
    # Config schema                                                    #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "seed": {**base["seed"], "help": "Seed of the roulette wheel of step 2, the only random component."},
            "theta": {
                "type": "int", "default": 40, "min": 0, "max": 100,
                "label": "Depth percentage (θ)",
                "help": "Depth β of step 1 as a percentage of the greedy solution's length, times a "
                        "factor that falls from 1.5 (up to 9 blocks) to 0.5 (25 blocks or more); the "
                        "paper prints these end values the other way round (L19). Paper: 40 (p. 448).",
            },
            "lam": {
                "type": "int", "default": 500, "min": 1, "max": 500,
                "label": "λ, nodes per subset N_g and N_q",
                "help": "Partial solutions kept from step 1 per subset (lowest g, lowest q). "
                        "Paper: 500 (p. 448).",
            },
            "node_budget": {
                "type": "int", "default": 10_000_000, "min": 1, "max": 1_000_000_000,
                "label": "Step-1 node budget",
                "help": "Nodes visited by step 1 over all iterative-deepening iterations; "
                        "not in the paper (L18).",
            },
            "rho": {
                "type": "int", "default": 300, "min": 1, "max": 10_000,
                "label": "Step-2 iterations (ρ)",
                "help": "Times every partial solution is completed. Paper: 300 (p. 448).",
            },
            "kappa": {
                "type": "int", "default": 2, "min": 1, "max": 20,
                "label": "Retrievals per expansion (κ)",
                "help": "Paper: 2 (p. 446).",
            },
            "mu": {
                "type": "int", "default": 5, "min": 1, "max": 100,
                "label": "Sequences in the roulette wheel (µ)",
                "help": "Paper: 5 (p. 446).",
            },
            "tau": {
                "type": "int", "default": 3, "min": 1, "max": 20,
                "label": "Branching factor of the expansion (τ)",
                "help": "Paper: 3 (p. 446).",
            },
            "time_limit_sec": {
                "type": "int", "default": 0, "min": 0, "max": 864_000,
                "label": "Time limit per instance (s)",
                "help": "0 = no limit, the paper's behaviour. Otherwise step 2 stops before the first and "
                        "after every completed partial solution once the limit has passed; the greedy "
                        "and step 1 are not interrupted. Not in the paper (L32).",
            },
        })
        # Not read by this method.
        for key in ("max_iterations", "report_interval"):
            base.pop(key)
        return base

    # ---------------------------------------------------------------- #
    # Train                                                            #
    # ---------------------------------------------------------------- #

    def train(
        self,
        problem_factory: Callable,
        result_queue: mp.Queue,
        stop_event: mp.Event,
    ) -> None:
        cfg = self.config
        rng_seed = cfg.seed
        n_seeds = 1  # multi-seed eval removed; single run only

        par = parameters(cfg, type(self))
        theta, lam, node_budget = int(par["theta"]), int(par["lam"]), int(par["node_budget"])
        rho, kappa, mu, tau = int(par["rho"]), int(par["kappa"]), int(par["mu"]), int(par["tau"])
        time_limit = float(par["time_limit_sec"])
        operating = cfg.extra.get("cost_model", "crane_time") == "operating_cost"
        weights = {k: float(cfg.extra[k]) for k in ("eta1", "gamma1", "eta2", "gamma2") if k in cfg.extra}

        all_metrics: List[Dict] = []

        for seed_idx in range(n_seeds):
            if stop_event.is_set():
                break

            # L32: the limit counts from here, so it includes the greedy and step 1 (not interrupted)
            deadline = time.monotonic() + time_limit if time_limit > 0 else None
            seed = rng_seed + seed_idx
            rng = random.Random(seed)   # the only random component: the roulette of step 2

            env = problem_factory()
            # Stack s of the paper is stack s - 1 of the bay; containers on top at the start stay in it.
            init_stacks, positions = read_bay(env)
            spec = ObjectiveSpec.from_config(env.config)
            check_objective(spec, self.name)

            # §4: f2 / f2vert from the CRP-Time time model (exact integers, L29); the paper's operating
            # cost only on request, for the validation in the paper's setting.
            cost = CostModel(n_T=env.config.max_tiers, **weights) if operating else TimeModel.from_spec(spec)
            root = Bay(init_stacks, cost)
            n_C = sum(len(st) for st in init_stacks)

            # ----- Step 1 (Sec. 6.1): the greedy (Sec. 7.1) gives ub and |phi| for beta (eq. 29),
            # Algorithm 1 runs to depth beta within the node budget (L18), and N = N_g + N_q -----
            greedy_phi, ub = greedy(root)
            depth = beta(len(greedy_phi), n_C, theta)
            partials, leaf, _, step1 = partial_solutions(root, ub, depth, lam, n_C, node_budget)

            # ----- Step 2 (Sec. 6.2, Alg. 2 lines 3-14): phi_best starts as the greedy solution (O20),
            # or the best step-1 leaf if cheaper (O05); rho iterations of Expansion from N -----
            phi_best, f_best = greedy_phi, ub
            if leaf is not None and leaf.g < f_best:
                phi_best, f_best = leaf.phi, leaf.g
            step2 = {"limit_hit": False, "stopped": False, "rho_done": 0, "partials_last": 0}
            phi_best, f_best = algorithm2_loop(partials, phi_best, f_best, rng, rho, kappa, tau, mu, n_C,
                                               deadline, step2, stop_event)
            if step2["stopped"]:
                return   # stopped by the user: no result, as when the stop comes before the run

            # ----- Step 3 (Sec. 6.3, Alg. 2 line 15): repair temporary movements. The plan is
            # phi_best itself; the problem validates it and reports its metrics -----
            try:
                phi_best, f_best = postprocess(init_stacks, phi_best, cost)
                f_plan = f(init_stacks, phi_best, cost)
            except ValueError as err:
                raise RuntimeError(f"López-Plata: impossible move in phi: {err}") from err
            plan = replay(init_stacks, positions, [(s - 1, s2 - 1 if s2 else None) for s, s2 in phi_best])
            best_solution = actions(plan, env.config.num_rows)
            final_metrics, env_final = finish(problem_factory, plan, spec, self.name)
            if operating:
                final_metrics["operating_cost"] = f_plan
            elif abs(cost.seconds(f_plan) - final_metrics["crane_time"]) > 1e-6:
                raise RuntimeError(f"López-Plata: own crane time {cost.seconds(f_plan)} differs from the "
                                   f"platform's {final_metrics['crane_time']}")
            final_metrics.update({
                "beta": float(depth),
                "beta_eff": float(step1["beta_eff"]),
                "budget_hit": float(step1["budget_hit"]),
                "visited_total": float(step1["visited_total"]),
                "time_limit_hit": float(step2["limit_hit"]),
                "rho_done": float(step2["rho_done"]),
                "partials_last": float(step2["partials_last"]),
            })
            all_metrics.append(final_metrics)

            primary = float(final_metrics["objective_value"])
            if primary < self._best_metric:
                self._best_metric = primary
                self._best_solution = list(best_solution)

            self._push(
                result_queue,
                step=seed_idx + 1,
                metric=primary,
                metrics=final_metrics,
                progress=(seed_idx + 1) / n_seeds,
                snapshot=env_final.get_state_snapshot() if hasattr(env_final, "get_state_snapshot") else None,
                extra={"operating_cost": f_plan} if operating else None,
            )

        if all_metrics:
            agg = {k: float(np.mean([m[k] for m in all_metrics if k in m]))
                   for k in all_metrics[0]}
            self._push(result_queue, step=n_seeds, metric=self._best_metric, metrics=agg, progress=1.0)

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution
