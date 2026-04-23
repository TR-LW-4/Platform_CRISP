"""
Ðurasević & Ðumić (2024) — GP hyper-heuristic for CRP-Time / BRP-Fixed.

Reference
---------
Ðurasević, M.; Ðumić, M. (2024)
"Designing relocation rules with genetic programming for the container
 relocation problem with multiple bays and container groups",
Applied Soft Computing 150: 111104.

Role on the platform (phase 1)
------------------------------
Covers the **multi-bay, distinct-due-date, restricted** variant
(paper §7.1, RS-only).  Same problem family as Lin 2015 / G-CREM 2020 /
Lee–Lee 2010, so this algorithm plugs straight into ``CRP-Time`` and
``BRP-Fixed`` with no modelling changes.

Two further variants are intentionally deferred:

* **Unrestricted RS** (paper Alg. 3) — needs an extended action space;
  to be added once ``BRP-Unrestricted`` / GUI support lands.
* **Container-groups RS** (paper §7.2) — needs a new ``CRP-Groups``
  problem class (duplicate due dates + origin-stack selection).

Training / evaluation protocol
------------------------------
* ``num_train_instances`` random yards (seeded deterministically from
  ``cfg.seed + 1_000``) form the training corpus.  The fitness of a GP
  individual is the **sum** of its RR's cost across these instances.
* Once GP terminates, the best tree is **re-evaluated independently**
  on ``num_eval_seeds`` freshly-seeded yards.  Those are the numbers
  reported through ``_push`` — fully consistent with the reporting
  convention of Lin 2015, G-CREM 2020 and the other baselines.

All paper hyperparameters are exposed via ``config_schema`` with
interactive-friendly defaults (smaller population / fewer evaluations
than the paper's 1000 × 50 000 to keep GUI runs under a minute).
"""

from __future__ import annotations

import copy
import multiprocessing as mp
import random
from typing import Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.objectives     import (
    KinematicsModel,
    compute_crane_time,
    lower_bound_relocations,
)
from core.plan           import RelocationPlan

from .gp_core         import Node, evolve_gp
from .rs_restricted   import build_plan_restricted
from .terminals       import build_terminal_table, terminal_names


# ================================================================ #
#  Helper: per-instance fitness                                      #
# ================================================================ #

def _plan_cost(
    plan:      RelocationPlan,
    kin:       KinematicsModel,
    objective: str,
) -> float:
    if objective == "crane_time":
        return float(compute_crane_time(plan, kin))
    return float(plan.num_relocations())


# ================================================================ #
#  Main algorithm                                                    #
# ================================================================ #

class Durasevic2024GP(BaseAlgorithm):

    name                = "Ðurasević–Ðumić (2024) GP"
    category            = "Heuristic"
    description         = (
        "[native multi-bay]  "
        "GP hyper-heuristic for CRP-Time (Ðurasević & Ðumić, ASOC 2024). "
        "Evolves an arithmetic-tree Priority Function that scores "
        "destination stacks; combined with the restricted RS, it beats "
        "manually designed rules by ~4–5 % on the Lee–Lee benchmark.  "
        "Terminals: SH/EMP/CUR/RI/AVG/DIFF (+ DIS/DUR for multi-bay). "
        "Phase-1 deployment: restricted RS, distinct due dates only."
    )
    compatible_problems = ["CRP-Time", "BRP-Fixed"]
    step_label          = "Seed"

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)
        self._best_plan: Optional[RelocationPlan] = None
        self._best_tree: Optional[Node]           = None

    # ---------------------------------------------------------------- #
    # Train                                                              #
    # ---------------------------------------------------------------- #

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        cfg       = self.config
        rng_seed  = cfg.seed
        n_seeds   = max(1, cfg.num_eval_seeds)

        # ── GP hyper-parameters (paper §6, interactive-friendly defaults) ── #
        pop_size      = int  (cfg.extra.get("pop_size",            50))
        max_depth     = int  (cfg.extra.get("max_depth",           5))
        max_evals     = int  (cfg.extra.get("max_evals",           2_000))
        mutation_prob = float(cfg.extra.get("mutation_prob",       0.3))
        n_train       = int  (cfg.extra.get("num_train_instances", 3))
        use_multibay  = bool (cfg.extra.get("use_multibay_extras", True))
        objective     = str  (cfg.extra.get("objective",           "auto"))

        # Resolve "auto" objective from the problem class name (CRP-Time family
        # minimises crane working time; BRP-Fixed family minimises relocations)
        if objective == "auto":
            probe_name = getattr(problem_factory(), "name", "")
            objective  = "crane_time" if "time" in probe_name.lower() else "relocations"

        # ── Build training corpus (deterministic) ────────────────── #
        training_instances: List[Dict] = []
        for t in range(n_train):
            env_t = problem_factory()
            env_t.config.seed = rng_seed + 1_000 + t
            env_t.reset()
            training_instances.append({
                "yard":       copy.deepcopy(env_t.yard),
                "containers": list(env_t.containers),
                "max_tiers":  env_t.config.max_tiers,
                "kin":        KinematicsModel.from_config_extra(env_t.config.extra),
            })

        terminal_table = build_terminal_table(use_multibay)
        term_names     = terminal_names(use_multibay)

        # ── Fitness function (summed over training instances) ────── #
        def fitness_fn(tree: Node) -> float:
            total = 0.0
            for ti in training_instances:
                try:
                    plan = build_plan_restricted(
                        ti["yard"], ti["containers"], tree,
                        terminal_table, ti["max_tiers"], ti["kin"],
                    )
                    total += _plan_cost(plan, ti["kin"], objective)
                except Exception:
                    # Degenerate tree / unreachable state → very bad fitness
                    return float("inf")
            return total

        # ── Evolve ──────────────────────────────────────────────── #
        rng = random.Random(rng_seed)

        report_pushes = {"n": 0}

        def report_cb(evals_done: int, best_fit: float, best_tree: Node) -> None:
            # Throttle GUI pushes — one every ~50 evals
            report_pushes["n"] += 1
            if report_pushes["n"] % 5 != 0 and evals_done < max_evals:
                return
            progress = min(0.5 * evals_done / max_evals, 0.5)
            self._push(
                result_queue,
                step     = 0,
                metric   = float(best_fit),
                metrics  = {
                    "gp_fitness":        float(best_fit),
                    "objective":         1.0 if objective == "crane_time" else 0.0,
                    "tree_size":         float(best_tree.size()),
                    "evals_done":        float(evals_done),
                },
                progress = progress,
                extra    = {"phase": "evolve"},
            )

        def stop_flag() -> bool:
            return stop_event.is_set()

        best_tree, best_fit = evolve_gp(
            fitness_fn     = fitness_fn,
            terminal_names = term_names,
            pop_size       = pop_size,
            max_depth      = max_depth,
            max_evals      = max_evals,
            mutation_prob  = mutation_prob,
            rng            = rng,
            report_cb      = report_cb,
            stop_flag      = stop_flag,
        )
        self._best_tree = best_tree

        # ── Evaluate on independent test seeds ──────────────────── #
        all_metrics: List[Dict] = []
        for seed_idx in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.config.seed = rng_seed + seed_idx
            env.reset()

            kin          = KinematicsModel.from_config_extra(env.config.extra)
            initial_yard = copy.deepcopy(env.yard)
            containers   = list(env.containers)
            max_tiers    = env.config.max_tiers
            lb           = lower_bound_relocations(initial_yard)

            plan = build_plan_restricted(
                initial_yard, containers, best_tree,
                terminal_table, max_tiers, kin,
            )

            n_relocs   = plan.num_relocations()
            crane_time = compute_crane_time(plan, kin)
            metrics = {
                "relocations": float(n_relocs),
                "crane_time":  float(crane_time),
                "total_moves": float(plan.num_moves()),
                "lower_bound": float(lb),
                "lb_ratio":    float(n_relocs / max(lb, 1)),
                "tree_size":   float(best_tree.size()),
                "time":        float(crane_time),
                "steps":       float(plan.num_moves()),
            }
            all_metrics.append(metrics)

            primary = float(crane_time if objective == "crane_time" else n_relocs)
            if primary < self._best_metric:
                self._best_metric   = primary
                self._best_plan     = plan
                self._best_solution = _plan_to_action_list(plan, env)

            self._push(
                result_queue,
                step     = seed_idx + 1,
                metric   = primary,
                metrics  = metrics,
                progress = 0.5 + 0.5 * (seed_idx + 1) / n_seeds,
                snapshot = env.get_state_snapshot(),
                extra    = {
                    "phase":      "test",
                    "seed":       seed_idx,
                    "objective":  objective,
                    "tree_repr":  repr(best_tree)[:200],
                },
            )

        if all_metrics:
            agg = {
                kname: float(np.mean([m[kname] for m in all_metrics if kname in m]))
                for kname in all_metrics[0]
            }
            self._push(
                result_queue,
                step     = n_seeds,
                metric   = self._best_metric,
                metrics  = agg,
                progress = 1.0,
            )

    # ---------------------------------------------------------------- #
    # Public accessors                                                   #
    # ---------------------------------------------------------------- #

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    def get_best_plan(self) -> Optional[RelocationPlan]:
        return self._best_plan

    def get_best_tree(self) -> Optional[Node]:
        return self._best_tree

    # ---------------------------------------------------------------- #
    # Config schema                                                      #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "num_eval_seeds": {
                "type": "int", "default": 3, "min": 1, "max": 100,
                "label": "Evaluation seeds",
                "help": "Test yards used to report the evolved PF's metrics.",
            },
            "num_train_instances": {
                "type": "int", "default": 3, "min": 1, "max": 50,
                "label": "Training yards",
                "help": "Random yards used to score each GP individual's fitness.",
            },
            "pop_size": {
                "type": "int", "default": 50, "min": 10, "max": 2000,
                "label": "GP population size",
                "help": "Paper uses 1000; 50 is a quick-preview default.",
            },
            "max_depth": {
                "type": "int", "default": 5, "min": 2, "max": 10,
                "label": "Tree depth cap",
                "help": "Paper's default is 5.",
            },
            "max_evals": {
                "type": "int", "default": 2000, "min": 100, "max": 200_000,
                "label": "Max GP evaluations",
                "help": "Total fitness evaluations. Paper uses 50000.",
            },
            "mutation_prob": {
                "type": "float", "default": 0.3, "min": 0.0, "max": 1.0,
                "label": "Subtree mutation prob.",
                "help": "Paper uses 0.3 for restricted RS.",
            },
            "use_multibay_extras": {
                "type": "bool", "default": True,
                "label": "Multi-bay terminals (DIS + DUR)",
                "help": "Append DIS and DUR terminals — enable on CRP-Time.",
            },
            "objective": {
                "type": "str", "default": "auto",
                "label": "GP objective",
                "help": (
                    "auto = pick by problem (CRP-Time → crane_time, "
                    "BRP-Fixed → relocations); or set explicitly to "
                    "'crane_time' / 'relocations'."
                ),
            },
        })
        return base


# ================================================================ #
#  Helper: plan → flat action list                                   #
# ================================================================ #

def _plan_to_action_list(plan: RelocationPlan, env) -> List[int]:
    num_rows = env.config.num_rows
    actions: List[int] = []
    for m in plan.movements:
        if m.to_pos is None:
            continue
        bay, row = m.to_pos
        actions.append((bay - 1) * num_rows + (row - 1))
    return actions
