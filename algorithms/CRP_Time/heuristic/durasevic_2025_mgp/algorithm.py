"""
Ðurasević–Ðumić–Gil-Gala (2025) — Multitask GP for CRP-Time.

Reference
---------
Ðurasević, M.; Ðumić, M.; Gil-Gala, F. J. (2025)
"Multitask genetic programming for automated design of heuristics for
 the container relocation problem",
Engineering Applications of Artificial Intelligence 144: 110001.

Role on the platform (phase 1)
------------------------------
Continuation of the 2024 single-task GP baseline.  This implementation
covers the paper's **four scenarios that do NOT require new problem
classes** — they can all be built from the existing ``CRP-Time`` /
``CRP-R`` families just by varying the ``ProblemConfig``:

* ``max_tiers``  (paper Scenario 1)  — subpops differ in stack-height cap.
* ``objective``  (paper Scenario 3)  — one subpop optimises relocations,
                                       another optimises crane_time; both
                                       train on the SAME yards.
                                       **Largest MGP gain** per Table 7.
* ``layout``     (paper Scenario 4 — simplified) — each subpop uses a
                                       different random seed for its
                                       training yards.
* ``load``       (paper Scenario 5)  — subpops differ in yard occupancy
                                       (``num_containers`` scaled).

The remaining two paper scenarios
(2: restricted vs unrestricted RS; 6: container-group duplicate ratios)
are deferred to phase 2 once ``BRP-Unrestricted`` and ``CRP-Groups``
problem classes are added.

Implementation reuse
--------------------
The GP tree, terminals, and restricted-RS plan builder are all reused
unchanged from ``algorithms.heuristic.durasevic_2024``.  The only new
piece is the **multi-subpopulation driver** in ``mgp.py``.
"""

from __future__ import annotations

import copy
import multiprocessing as mp
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.base_problem   import ProblemConfig
from core.objectives     import (
    KinematicsModel,
    compute_crane_time,
    lower_bound_relocations,
)
from core.plan           import RelocationPlan

from core.gp.engine        import Node
from core.gp.terminals     import (
    build_terminal_table,
    terminal_names,
)
from core.gp.rs_restricted import (
    build_plan_restricted,
)

from .mgp import Task, SubPopState, evolve_mgp


# ================================================================ #
#  Shared fitness helper                                              #
# ================================================================ #

def _plan_cost(plan: RelocationPlan, kin: KinematicsModel, objective: str) -> float:
    if objective == "crane_time":
        return float(compute_crane_time(plan, kin))
    return float(plan.num_relocations())


def _build_training_instance(ProblemCls, cfg: ProblemConfig) -> Dict:
    """Instantiate + reset a single training yard, capture its state."""
    env = ProblemCls(copy.deepcopy(cfg))
    env.reset()
    return {
        "yard":       copy.deepcopy(env.yard),
        "containers": list(env.containers),
        "max_tiers":  env.config.max_tiers,
        "kin":        KinematicsModel.from_config_extra(env.config.extra),
    }


def _make_fitness_fn(
    training_instances: List[Dict],
    terminal_table:     Dict[str, Callable],
    objective:          str,
) -> Callable[[Node], float]:
    """Sum-of-cost fitness across the training corpus."""
    def _fitness(tree: Node) -> float:
        total = 0.0
        for ti in training_instances:
            try:
                plan = build_plan_restricted(
                    ti["yard"], ti["containers"], tree,
                    terminal_table, ti["max_tiers"], ti["kin"],
                )
                total += _plan_cost(plan, ti["kin"], objective)
            except Exception:
                return float("inf")
        return total
    return _fitness


# ================================================================ #
#  Scenario task builders                                             #
# ================================================================ #

def _build_tasks_max_tiers(
    ProblemCls,
    base_cfg:       ProblemConfig,
    num_subpops:    int,
    n_train:        int,
    seed_base:      int,
    terminal_table: Dict[str, Callable],
) -> List[Task]:
    """Paper §6.1 — subpops differ in maximum stack height (H = 6, 8, ...)."""
    mt_schedule = [6, 8, 10, 12][:num_subpops] or [base_cfg.max_tiers]
    if len(mt_schedule) < num_subpops:
        # Fill with base value if user wants more than 4 subpops.
        mt_schedule = mt_schedule + [base_cfg.max_tiers] * (num_subpops - len(mt_schedule))

    tasks: List[Task] = []
    for i, mt in enumerate(mt_schedule):
        task_cfg           = copy.deepcopy(base_cfg)
        task_cfg.max_tiers = mt
        training = [
            _build_training_instance(
                ProblemCls,
                _with_seed(task_cfg, seed_base + 1000 + i * 100 + t),
            )
            for t in range(n_train)
        ]
        tasks.append(Task(
            name       = f"max_tiers={mt}",
            fitness_fn = _make_fitness_fn(training, terminal_table, "crane_time"),
            meta       = {"test_cfg": task_cfg, "objective": "crane_time"},
        ))
    return tasks


def _build_tasks_objective(
    ProblemCls,
    base_cfg:       ProblemConfig,
    num_subpops:    int,
    n_train:        int,
    seed_base:      int,
    terminal_table: Dict[str, Callable],
) -> List[Task]:
    """Paper §6.3 — two subpops on the SAME yards: relocations vs crane_time."""
    # SP forced to 2 in this scenario (remaining slots repeat).
    objectives = ["relocations", "crane_time"]
    if num_subpops > 2:
        objectives += [objectives[i % 2] for i in range(num_subpops - 2)]
    objectives = objectives[:num_subpops]

    # Shared training yards across all subpops.
    shared_training = [
        _build_training_instance(
            ProblemCls,
            _with_seed(base_cfg, seed_base + 1000 + t),
        )
        for t in range(n_train)
    ]

    tasks: List[Task] = []
    for i, obj in enumerate(objectives):
        tasks.append(Task(
            name       = f"objective={obj}",
            fitness_fn = _make_fitness_fn(shared_training, terminal_table, obj),
            meta       = {"test_cfg": copy.deepcopy(base_cfg), "objective": obj},
        ))
    return tasks


def _build_tasks_layout(
    ProblemCls,
    base_cfg:       ProblemConfig,
    num_subpops:    int,
    n_train:        int,
    seed_base:      int,
    terminal_table: Dict[str, Callable],
) -> List[Task]:
    """Paper §6.4 (simplified) — each subpop gets its own random-seed block for
    training yards, so the evolved rules learn from different instances."""
    tasks: List[Task] = []
    for i in range(num_subpops):
        # Distinct seed block per subpop → different yard draws.
        training = [
            _build_training_instance(
                ProblemCls,
                _with_seed(base_cfg, seed_base + 10_000 * (i + 1) + t),
            )
            for t in range(n_train)
        ]
        tasks.append(Task(
            name       = f"layout#{i}",
            fitness_fn = _make_fitness_fn(training, terminal_table, "crane_time"),
            meta       = {"test_cfg": copy.deepcopy(base_cfg), "objective": "crane_time"},
        ))
    return tasks


def _build_tasks_load(
    ProblemCls,
    base_cfg:       ProblemConfig,
    num_subpops:    int,
    n_train:        int,
    seed_base:      int,
    terminal_table: Dict[str, Callable],
) -> List[Task]:
    """Paper §6.5 — subpops differ in yard occupancy (fill rate)."""
    capacity = base_cfg.num_bays * base_cfg.num_rows * base_cfg.max_tiers
    loads    = [0.5, 0.7, 0.9, 0.3][:num_subpops]
    if len(loads) < num_subpops:
        loads = loads + [0.7] * (num_subpops - len(loads))

    tasks: List[Task] = []
    for i, rho in enumerate(loads):
        task_cfg                = copy.deepcopy(base_cfg)
        task_cfg.num_containers = max(1, int(round(rho * capacity)))
        training = [
            _build_training_instance(
                ProblemCls,
                _with_seed(task_cfg, seed_base + 1000 + i * 100 + t),
            )
            for t in range(n_train)
        ]
        tasks.append(Task(
            name       = f"load={rho:.2f}",
            fitness_fn = _make_fitness_fn(training, terminal_table, "crane_time"),
            meta       = {"test_cfg": task_cfg, "objective": "crane_time"},
        ))
    return tasks


def _with_seed(cfg: ProblemConfig, seed: int) -> ProblemConfig:
    new = copy.deepcopy(cfg)
    new.seed = int(seed)
    return new


_SCENARIOS: Dict[str, Callable] = {
    "max_tiers": _build_tasks_max_tiers,
    "objective": _build_tasks_objective,
    "layout":    _build_tasks_layout,
    "load":      _build_tasks_load,
}


# ================================================================ #
#  Main algorithm                                                      #
# ================================================================ #

class Durasevic2025MGP(BaseAlgorithm):

    name                = "Ðurasević–Ðumić–Gil-Gala (2025) MGP"
    category            = "Heuristic"
    description         = (
        "[native multi-bay]  "
        "Multitask GP for CRP-Time (Ðurasević, Ðumić & Gil-Gala, EAAI 2025). "
        "Evolves Priority Functions for several CRP tasks simultaneously "
        "in separate subpopulations that share knowledge via cross-subpop "
        "crossover and/or ring-topology migration.  Scenarios covered: "
        "max_tiers / objective / layout / load.  The 'objective' scenario "
        "reproduces the paper's Table 7 gain on crane_time."
    )
    compatible_problems = ["CRP-Time"]
    step_label          = "Subpop-seed"

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)
        self._best_plan: Optional[RelocationPlan] = None
        self._best_tree: Optional[Node]           = None
        self._subpop_best_trees: Dict[str, Node]  = {}

    # ---------------------------------------------------------------- #
    # Train                                                              #
    # ---------------------------------------------------------------- #

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        cfg      = self.config
        rng_seed = cfg.seed
        n_seeds  = max(1, cfg.num_eval_seeds)

        # ── Hyper-parameters ─────────────────────────────────────── #
        scenario        = str (cfg.extra.get("scenario",            "objective"))
        num_subpops     = int (cfg.extra.get("num_subpopulations",  2))
        transfer_mode   = str (cfg.extra.get("transfer_mode",       "both"))
        p_ct            = float(cfg.extra.get("p_ct",               0.5))
        migration_M     = int (cfg.extra.get("migration_M",         20))
        migration_freq  = int (cfg.extra.get("migration_frequency", 500))

        pop_per_sub     = int (cfg.extra.get("pop_size_per_subpop", 30))
        max_depth       = int (cfg.extra.get("max_depth",           5))
        max_evals       = int (cfg.extra.get("max_evals",           2_000))
        mutation_prob   = float(cfg.extra.get("mutation_prob",      0.3))
        n_train         = int (cfg.extra.get("num_train_instances", 3))

        # Sanity
        num_subpops = max(2, min(num_subpops, 5))
        if scenario == "objective":
            num_subpops = 2  # paper spec for §6.3 (relocations vs crane_time)

        if scenario not in _SCENARIOS:
            raise ValueError(
                f"Unknown scenario '{scenario}'. "
                f"Allowed: {sorted(_SCENARIOS.keys())}"
            )

        # ── Reference problem class + config ─────────────────────── #
        ref_env     = problem_factory()
        ProblemCls  = type(ref_env)
        base_cfg    = copy.deepcopy(ref_env.config)

        # ── Build tasks (one per subpop) ─────────────────────────── #
        terminal_table = build_terminal_table(use_multibay_extras=True)
        term_names     = terminal_names(use_multibay_extras=True)

        tasks = _SCENARIOS[scenario](
            ProblemCls, base_cfg, num_subpops, n_train, rng_seed, terminal_table,
        )

        # ── Progress reporter ────────────────────────────────────── #
        last_push = {"n": 0}

        def report_cb(evals_done: int, subpops: List[SubPopState]) -> None:
            last_push["n"] += 1
            if last_push["n"] % 3 != 0 and evals_done < max_evals:
                return
            best_per_sub = {
                f"sub{i}_fit": float(sp.best_fitness)
                for i, sp in enumerate(subpops)
            }
            self._push(
                result_queue,
                step     = 0,
                metric   = float(min(sp.best_fitness for sp in subpops)),
                metrics  = {
                    **best_per_sub,
                    "evals_done": float(evals_done),
                    "mean_tree_size": float(np.mean([sp.best_tree.size()
                                                    for sp in subpops])),
                },
                progress = min(0.5 * evals_done / max_evals, 0.5),
                extra    = {
                    "phase": "evolve",
                    "scenario": scenario,
                    "subpop_tasks": [sp.task.name for sp in subpops],
                },
            )

        def stop_flag() -> bool:
            return stop_event.is_set()

        # ── Evolve ──────────────────────────────────────────────── #
        rng = random.Random(rng_seed)
        subpops = evolve_mgp(
            tasks               = tasks,
            terminal_names      = term_names,
            pop_size_per_subpop = pop_per_sub,
            max_depth           = max_depth,
            max_evals           = max_evals,
            mutation_prob       = mutation_prob,
            transfer_mode       = transfer_mode,
            p_ct                = p_ct,
            migration_M         = migration_M,
            migration_frequency = migration_freq,
            rng                 = rng,
            report_cb           = report_cb,
            stop_flag           = stop_flag,
        )

        # ── Evaluate each subpop's best tree on its OWN test yards ── #
        all_metrics: List[Dict] = []
        total_sub_seed          = num_subpops * n_seeds
        step_counter            = 0

        for sp_idx, sp in enumerate(subpops):
            best_tree = sp.best_tree
            self._subpop_best_trees[sp.task.name] = best_tree

            test_cfg   = sp.task.meta.get("test_cfg", base_cfg)
            sub_obj    = sp.task.meta.get("objective", "crane_time")

            for seed_idx in range(n_seeds):
                if stop_flag():
                    break
                env                 = ProblemCls(copy.deepcopy(test_cfg))
                env.config.seed     = rng_seed + seed_idx
                env.reset()

                kin          = KinematicsModel.from_config_extra(env.config.extra)
                initial_yard = copy.deepcopy(env.yard)
                containers   = list(env.containers)
                mt           = env.config.max_tiers
                lb           = lower_bound_relocations(initial_yard)

                plan = build_plan_restricted(
                    initial_yard, containers, best_tree,
                    terminal_table, mt, kin,
                )
                n_relocs   = plan.num_relocations()
                crane_time = compute_crane_time(plan, kin)

                metrics = {
                    "relocations":  float(n_relocs),
                    "crane_time":   float(crane_time),
                    "total_moves":  float(plan.num_moves()),
                    "lower_bound":  float(lb),
                    "lb_ratio":     float(n_relocs / max(lb, 1)),
                    "tree_size":    float(best_tree.size()),
                    "time":         float(crane_time),
                    "steps":        float(plan.num_moves()),
                }
                all_metrics.append(metrics)

                primary = float(
                    crane_time if sub_obj == "crane_time" else n_relocs
                )
                if primary < self._best_metric:
                    self._best_metric   = primary
                    self._best_plan     = plan
                    self._best_tree     = best_tree
                    self._best_solution = _plan_to_action_list(plan, env)

                step_counter += 1
                self._push(
                    result_queue,
                    step     = step_counter,
                    metric   = primary,
                    metrics  = metrics,
                    progress = 0.5 + 0.5 * step_counter / max(1, total_sub_seed),
                    snapshot = env.get_state_snapshot(),
                    extra    = {
                        "phase":     "test",
                        "subpop":    sp.task.name,
                        "seed":      seed_idx,
                        "objective": sub_obj,
                        "tree_repr": repr(best_tree)[:200],
                    },
                )

        # ── Final aggregate ──────────────────────────────────────── #
        if all_metrics:
            agg = {
                kname: float(np.mean([m[kname] for m in all_metrics if kname in m]))
                for kname in all_metrics[0]
            }
            self._push(
                result_queue,
                step     = step_counter or total_sub_seed,
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

    def get_subpop_best_trees(self) -> Dict[str, Node]:
        return dict(self._subpop_best_trees)

    # ---------------------------------------------------------------- #
    # Config schema                                                      #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "num_eval_seeds": {
                "type": "int", "default": 2, "min": 1, "max": 50,
                "label": "Evaluation seeds",
                "help": "Test yards per subpop used for reporting metrics.",
            },
            "scenario": {
                "type": "str", "default": "objective",
                "label": "MGP scenario",
                "help": (
                    "Which of the paper's scenarios to instantiate: "
                    "'max_tiers' (§6.1), 'objective' (§6.3, largest gain), "
                    "'layout' (§6.4, simplified), 'load' (§6.5)."
                ),
            },
            "num_subpopulations": {
                "type": "int", "default": 2, "min": 2, "max": 5,
                "label": "Number of subpopulations (S_P)",
                "help": "Forced to 2 for the 'objective' scenario.",
            },
            "transfer_mode": {
                "type": "str", "default": "both",
                "label": "Knowledge transfer mode",
                "help": "'crossover' | 'migration' | 'both' | 'none'.",
            },
            "p_ct": {
                "type": "float", "default": 0.5, "min": 0.0, "max": 1.0,
                "label": "Cross-subpop crossover prob (p_ct)",
                "help": "Chance that parent 2 comes from a different subpop.",
            },
            "migration_M": {
                "type": "int", "default": 20, "min": 1, "max": 1000,
                "label": "Migrants per transfer (M)",
                "help": "Number of individuals migrated between subpops each "
                        "time the migration operator fires.",
            },
            "migration_frequency": {
                "type": "int", "default": 500, "min": 1, "max": 100_000,
                "label": "Migration frequency (evals)",
                "help": "Fitness evaluations between successive migrations.",
            },
            "pop_size_per_subpop": {
                "type": "int", "default": 30, "min": 10, "max": 1000,
                "label": "Subpop size",
                "help": "Paper uses ~500 (= 1000 total / 2 subpops).",
            },
            "max_depth": {
                "type": "int", "default": 5, "min": 2, "max": 10,
                "label": "Tree depth cap",
            },
            "max_evals": {
                "type": "int", "default": 2000, "min": 100, "max": 200_000,
                "label": "Max GP evaluations",
                "help": "Total fitness evaluations across ALL subpops. "
                        "Paper uses 50 000.",
            },
            "mutation_prob": {
                "type": "float", "default": 0.3, "min": 0.0, "max": 1.0,
                "label": "Subtree mutation prob.",
            },
            "num_train_instances": {
                "type": "int", "default": 3, "min": 1, "max": 50,
                "label": "Training yards per subpop",
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
