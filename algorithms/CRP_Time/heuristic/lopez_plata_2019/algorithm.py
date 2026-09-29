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
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.plan import Movement, RelocationPlan

from .measures import beta, greedy, q
from .operating_cost import Bay, CostModel, Move, f
from .step1 import algorithm1, select


# ================================================================ #
#  Operating cost model (paper style)                               #
# ================================================================ #

@dataclass
class OperatingCostModel:
    """
    Linear operating cost for crane moves (López-Plata et al. 2019).

    Unloaded move cost from (s1, t1) to (s2, t2):
        m = (nT + t1)*α1 + d(s1,s2)*β1 + (nT + t2)*α1

    Loaded move cost:
        m' = (nT + t1)*α2 + d(s1,s2)*β2 + (nT + t2)*α2

    Retrieval point is modelled as stack 0, tier 1.
    """
    alpha1: float = 1.0   # unloaded vertical
    beta1: float = 1.0    # unloaded horizontal
    alpha2: float = 2.0   # loaded vertical
    beta2: float = 2.0    # loaded horizontal
    security_margin: int = 2  # extra tiers when raising the spreader

    def unloaded_cost(self, nT: int, t1: int, t2: int, dist: float) -> float:
        h = nT + self.security_margin
        return (h + t1) * self.alpha1 + dist * self.beta1 + (h + t2) * self.alpha1

    def loaded_cost(self, nT: int, t1: int, t2: int, dist: float) -> float:
        h = nT + self.security_margin
        return (h + t1) * self.alpha2 + dist * self.beta2 + (h + t2) * self.alpha2

    def retrieval_cost(self, nT: int, t: int) -> float:
        """Cost to retrieve from tier t (loaded) to retrieval point (0,1)."""
        return self.loaded_cost(nT, t, 1, 0.0)  # dist 0 for dummy stack 0

    def crane_to_pickup_unloaded(self, nT: int, from_top: int, to_t: int) -> float:
        """Cost to move unloaded crane from top of current stack down to a tier."""
        h = nT + self.security_margin
        # From top (h) to the slot tier to_t (the paper uses the slot tier directly)
        return (h + from_top) * self.alpha1 + 0.0 * self.beta1 + (h + to_t) * self.alpha1


def stack_distance(s1: int, s2: int) -> float:
    """Simple linear stack distance (paper default in experiments)."""
    return abs(s1 - s2)


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
#  Old step 2 (replaced in step 4), working on Bay nodes            #
# ================================================================ #

def _retrieve_on_top(bay: Bay) -> None:
    """Retrieve c* while it is on top of its stack."""
    while bay.target is not None:
        s = bay.target_stack()
        if bay.stacks[s - 1][-1] != bay.target:
            return
        bay.apply((s, 0))


def _relocation_quality(bay: Bay, src: int, dst: int, cost_model: OperatingCostModel, n_C: int,
                        w_move: float = 1.0, w_b: float = 1.0) -> float:
    """Old step-2 score of relocating the top of src to dst (stacks 1-based): the old move cost,
    normalised by a large move, plus w_b times q of the resulting configuration (measures.q)."""
    nT = bay.cost.n_T
    move_c = cost_model.loaded_cost(nT, len(bay.stacks[src - 1]), len(bay.stacks[dst - 1]) + 1,
                                    stack_distance(src, dst))
    tmp = bay.copy()
    tmp.apply((src, dst))
    max_move = cost_model.loaded_cost(nT, nT, nT, max(1, len(bay.stacks) - 1))
    return w_move * move_c / max_move + w_b * q(tmp, n_C)


def _expand_partial(bay: Bay, cost_model: OperatingCostModel, mu: int,
                    n_C: int) -> Tuple[List[int], float, Bay]:
    """Old step 2: mu steps, each retrieving c* or relocating c+ to the stack with the lowest
    _relocation_quality and then retrieving what became accessible. Returns the destinations,
    their old relocation cost and the resulting node."""
    current = bay.copy()
    nT = current.cost.n_T
    sequence_cost, dests = 0.0, []
    for _ in range(mu):
        if current.target is None:
            break
        s = current.target_stack()
        if current.stacks[s - 1][-1] == current.target:
            current.apply((s, 0))
            continue
        cands = [(_relocation_quality(current, s, d, cost_model, n_C), d)
                 for d in range(1, len(current.stacks) + 1)
                 if d != s and len(current.stacks[d - 1]) < nT]
        if not cands:
            break
        _, d = min(cands)
        sequence_cost += cost_model.loaded_cost(nT, len(current.stacks[s - 1]), len(current.stacks[d - 1]) + 1,
                                                stack_distance(s, d))
        current.apply((s, d))
        dests.append(d)
        _retrieve_on_top(current)
    return dests, sequence_cost, current


# ================================================================ #
#  Post-processing: remove temporary moves                          #
# ================================================================ #

def _remove_temporary_moves(phi: List[Move]) -> List[Move]:
    """Placeholder for step 3 of the paper (Sec. 6.3); returns phi unchanged."""
    return list(phi)


# ================================================================ #
#  Main heuristic class                                             #
# ================================================================ #

class LopezPlata2019Heuristic(BaseAlgorithm):
    """
    Heuristic based on A* for minimizing operating cost (López-Plata et al., C&IE 2019).

    Parameters via config.extra:
        theta (float): percentage in eq. 29 for the depth beta of step 1 (default 40).
        max_partial_nodes (int): lambda, nodes per subset N_g and N_q (default 500).
        num_expansions (int): how many times to repeat the old expansion phase (default 5).
        mu (int): steps per call of the old expansion (default 2).
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
            "num_expansions": {
                "type": "int", "default": 5, "min": 1, "max": 50,
                "label": "Number of expansion repetitions",
            },
            "mu": {
                "type": "int", "default": 2, "min": 1, "max": 10,
                "label": "Blocks per expansion (μ)",
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
        num_exp = int(cfg.extra.get("num_expansions", 5))
        mu = int(cfg.extra.get("mu", 2))

        # Old step-2 cost model, used only by the old expansion until step 4.
        cost_model = OperatingCostModel(
            alpha1=float(cfg.extra.get("alpha1", 1.0)),
            beta1=float(cfg.extra.get("beta1", 1.0)),
            alpha2=float(cfg.extra.get("alpha2", 2.0)),
            beta2=float(cfg.extra.get("beta2", 2.0)),
        )

        all_metrics: List[Dict] = []

        for seed_idx in range(n_seeds):
            if stop_event.is_set():
                break

            seed = rng_seed + seed_idx
            np.random.seed(seed)

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
            # Algorithm 1 runs to depth beta, and N = N_g + N_q -----
            greedy_phi, ub = greedy(root)
            nodes, leaf, _ = algorithm1(root, ub, beta(len(greedy_phi), n_C, theta))
            partials = select(nodes, lam, n_C)

            # phi_best starts as the greedy solution (O20); the best leaf of step 1 is a candidate (O05).
            candidates = [greedy_phi] + ([leaf.phi] if leaf else [])
            if partials:
                best_cost = float("inf")
                for _exp in range(num_exp):
                    phi_for_this, op_cost_for_this = self._construct_solution(
                        partials, cost_model, num_exp, mu, n_C)
                    # Selection among the expansions still on the old step-2 relocation cost (C15).
                    if op_cost_for_this < best_cost:
                        best_cost = op_cost_for_this
                        expanded = phi_for_this
                candidates.append(expanded)

            # phi_best has the lowest f (Alg. 2, line 10; ties keep the earlier candidate). The plan is
            # phi_best itself; the problem validates it and reports its metrics.
            try:
                best_phi = min(candidates, key=lambda phi: f(init_stacks, phi, cost))
                operating_cost, plan, best_solution = _plan_from_phi(init_stacks, best_phi, cost, n_rows)
            except ValueError as err:
                raise RuntimeError(f"López-Plata: impossible move in phi: {err}") from err
            env_final = problem_factory()
            if not env_final.validate_plan(plan).get("feasible"):
                raise RuntimeError("López-Plata: plan rejected by validate_plan: "
                                   + "; ".join(env_final.get_last_validation_errors()))

            final_metrics = env_final.get_metrics()
            final_metrics["operating_cost"] = operating_cost
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
    # Helper: construct a full solution using the heuristic            #
    # ---------------------------------------------------------------- #

    def _construct_solution(
        self,
        partials: List[Bay],
        cost_model: OperatingCostModel,
        num_exp: int,
        mu: int,
        n_C: int,
    ) -> Tuple[List[Move], float]:
        """
        Run the old expansion (replaced in step 4) from the partial solutions of step 1, round
        robin, and return phi (all moves from the root) + the old step-2 relocation cost.
        """
        best_phi: List[Move] = []
        best_cost = float("inf")

        for i in range(num_exp):
            cstate = partials[i % len(partials)].copy()
            _retrieve_on_top(cstate)   # the old expansion stops on a round without relocation
            total_c = 0.0

            while cstate.target is not None:
                seq, sc, cstate = _expand_partial(cstate, cost_model, mu, n_C)
                total_c += sc
                if not seq:
                    break

            cleaned = _remove_temporary_moves(cstate.phi)

            if total_c < best_cost:
                best_cost = total_c
                best_phi = cleaned

        return best_phi, best_cost

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution
