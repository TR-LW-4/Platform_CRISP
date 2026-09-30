"""
LopezPlata2019Heuristic
<2019> <heuristic> <time> <single-bay> <CRP-Time>
Limited-depth A* heuristic for operating-cost BRP
delta --- 3 --- Partial-search depth
num_expansions --- 5 --- Expansion count in Step 2

------------------------------- Reference --------------------------------
I. López-Plata, C. Expósito-Izquierdo, J.M. Moreno-Vega,
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
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.plan import Movement, RelocationPlan

from .measures import beta, greedy, p
from .operating_cost import Bay, CostModel, Move
from .step1 import partial_solutions
from .step2 import algorithm2_loop
from .step3 import postprocess


# ================================================================ #
#  Platform plan                                                    #
# ================================================================ #

def _plan_from_phi(stacks: List[List[int]], phi: List[Move], cost: CostModel,
                   n_rows: int) -> Tuple[float, RelocationPlan, List[int]]:
    """Replay phi through Bay (impossible moves raise ValueError). Return f(phi) (eq. 3),
    the platform plan and the env actions (0-based destination of each relocation).
    Stack s is flat index s - 1, i.e. (bay, row) with bays counted from the retrieval point."""
    def pos(s: int) -> Tuple[int, int]:
        return (s - 1) // n_rows + 1, (s - 1) % n_rows + 1

    bay = Bay(stacks, cost)
    plan, actions, f_phi = RelocationPlan(), [], 0.0
    for s, s2 in phi:
        stack = bay.stacks[s - 1] if 1 <= s <= len(bay.stacks) else []
        block = stack[-1] if stack else None
        f_phi += bay.apply((s, s2))
        plan.add(Movement(block, pos(s), None if s2 == 0 else pos(s2)))
        if s2:
            actions.append(s2 - 1)
    return f_phi, plan, actions


# ================================================================ #
#  Main heuristic class                                             #
# ================================================================ #

class LopezPlata2019Heuristic(BaseAlgorithm):
    """
    Heuristic based on A* for minimizing operating cost (López-Plata et al., C&IE 2019).

    Parameters via config.extra:
        theta (float): percentage in eq. 29 for the depth beta of step 1 (default 40).
        max_partial_nodes (int): lambda, nodes per subset N_g and N_q (default 500).
        node_budget (int): nodes visited by step 1 over all iterations (default 10**7, L18).
        rho (int): iterations of step 2 (default 300).
        kappa (int): retrievals per Expansion (default 2).
        mu (int): sequences in the roulette wheel (default 5).
        tau (int): branching factor of Expansion (default 3).
    """

    name = "López-Plata et al. (2019) Operating-Cost Heuristic"
    category = "Heuristic"
    description = (
        "A* based heuristic for the Blocks Relocation Problem with Operating Costs (BRP-OC), "
        "López-Plata et al. (C&IE 2019). "
        "Uses the paper's own linear operating cost model (configurable α/β weights) as the primary objective. "
        "Step 1: limited-depth A* to find promising partial solutions. "
        "Step 2: expansion using composite quality (move cost + yard quality). "
        "Step 3: post-processing to remove temporary moves. "
        "Only available for CRP-Time. The algorithm is self-contained and does not depend on "
        "other CRP-Time heuristics or the Lee-Lee kinematics model for its decisions."
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
            "delta": {
                "type": "int", "default": 3, "min": 1, "max": 10,
                "label": "A* partial depth (δ)",
            },
            "max_partial_nodes": {
                "type": "int", "default": 50, "min": 5, "max": 500,
                "label": "Max promising partial nodes",
            },
            "rho": {
                "type": "int", "default": 300, "min": 1, "max": 10_000,
                "label": "Step-2 iterations (ρ)",
            },
            "kappa": {
                "type": "int", "default": 2, "min": 1, "max": 20,
                "label": "Retrievals per expansion (κ)",
            },
            "mu": {
                "type": "int", "default": 5, "min": 1, "max": 100,
                "label": "Sequences in the roulette wheel (µ)",
            },
            "tau": {
                "type": "int", "default": 3, "min": 1, "max": 20,
                "label": "Branching factor of the expansion (τ)",
            },
            "node_budget": {
                "type": "int", "default": 10_000_000, "min": 1, "max": 1_000_000_000,
                "label": "Step-1 node budget",
                "help": "Nodes visited by step 1 over all iterative-deepening iterations; "
                        "not in the paper (L18).",
            },
            "alpha1": {"type": "float", "default": 1.0, "label": "Unloaded vertical weight"},
            "beta1":  {"type": "float", "default": 1.0, "label": "Unloaded horizontal weight"},
            "alpha2": {"type": "float", "default": 2.0, "label": "Loaded vertical weight"},
            "beta2":  {"type": "float", "default": 2.0, "label": "Loaded horizontal weight"},
        })
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

        theta = float(cfg.extra.get("theta", 40))
        lam = int(cfg.extra.get("max_partial_nodes", 500))
        node_budget = int(cfg.extra.get("node_budget", 10 ** 7))
        rho = int(cfg.extra.get("rho", 300))
        kappa = int(cfg.extra.get("kappa", 2))
        mu = int(cfg.extra.get("mu", 5))
        tau = int(cfg.extra.get("tau", 3))

        all_metrics: List[Dict] = []

        for seed_idx in range(n_seeds):
            if stop_event.is_set():
                break

            seed = rng_seed + seed_idx
            rng = random.Random(seed)   # the only random component: the roulette of step 2

            env = problem_factory()
            # Containers on top at the start stay in the yard, so their retrievals are part of phi.
            env.reset(options={"skip_auto_retrieve": True})

            n_bays = env.config.num_bays
            n_rows = env.config.num_rows
            nT = env.config.max_tiers
            nS = n_bays * n_rows

            # Initial stacks (priorities bottom->top)
            init_stacks: List[List[int]] = [[] for _ in range(nS)]
            for (bay, row), stk in env.yard.stacks.items():
                flat = (bay - 1) * n_rows + (row - 1)
                for c in stk.containers:   # bottom to top
                    init_stacks[flat].append(int(c.priority))

            cost = CostModel(n_T=nT)
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
            phi_best, f_best = self._step2(partials, phi_best, f_best, rng, rho, kappa, tau, mu, n_C)

            # ----- Step 3 (Sec. 6.3, Alg. 2 line 15): repair temporary movements. The plan is
            # phi_best itself; the problem validates it and reports its metrics -----
            try:
                phi_best, f_best = postprocess(init_stacks, phi_best, cost)
                operating_cost, plan, best_solution = _plan_from_phi(init_stacks, phi_best, cost, n_rows)
            except ValueError as err:
                raise RuntimeError(f"López-Plata: impossible move in phi: {err}") from err
            env_final = problem_factory()
            if not env_final.validate_plan(plan).get("feasible"):
                raise RuntimeError("López-Plata: plan rejected by validate_plan: "
                                   + "; ".join(env_final.get_last_validation_errors()))

            final_metrics = env_final.get_metrics()
            final_metrics["operating_cost"] = operating_cost
            # Step 1 under the budget: beta = floor(theta * scale) (eq. 29), so beta_eff holds for
            # every theta in [beta_eff / scale, (beta_eff + 1) / scale).
            scale = float(len(greedy_phi) * p(n_C) / 100)
            final_metrics.update({
                "beta": float(depth),
                "beta_eff": float(step1["beta_eff"]),
                "theta_eff_min": step1["beta_eff"] / scale,
                "theta_eff_max": (step1["beta_eff"] + 1) / scale,
                "budget_hit": float(step1["budget_hit"]),
                "visited_total": float(step1["visited_total"]),
                "visited_last": float(step1["visited_last"]),
                "n_beta": float(step1["n_beta"]),
            })
            all_metrics.append(final_metrics)

            # Keep the paper's operating cost as a reported diagnostic, while
            # platform-wide ranking follows the user-selected CRP-Time objective.
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
                extra={"operating_cost": operating_cost},
            )

        if all_metrics:
            agg = {k: float(np.mean([m[k] for m in all_metrics if k in m]))
                   for k in all_metrics[0]}
            self._push(result_queue, step=n_seeds, metric=self._best_metric, metrics=agg, progress=1.0)

    # ---------------------------------------------------------------- #
    # Step 2                                                           #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _step2(partials: List[Bay], phi_best: List[Move], f_best: float, rng: random.Random,
               rho: int, kappa: int, tau: int, mu: int, n_C: int) -> Tuple[List[Move], float]:
        return algorithm2_loop(partials, phi_best, f_best, rng, rho, kappa, tau, mu, n_C)

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution
