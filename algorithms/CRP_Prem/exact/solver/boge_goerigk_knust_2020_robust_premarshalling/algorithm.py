"""
Boge, Goerigk, Knust (2020) robust optimization for pre-marshalling.

Paper:
S. Boge, M. Goerigk, S. Knust,
"Robust optimization for premarshalling with uncertain priority classes",
European Journal of Operational Research 287 (2020) 191-210.

Embedding scope (v1):
  - uncertainty model via adjacent swaps (Kendall distance parameter delta)
  - theorem-based robust-existence checks (paper Sections 3.1/3.2)
  - solver-based optimization of BIbar^rob upper-bound objective
  - optional exact BI^rob evaluation by explicit scenario enumeration
"""

from __future__ import annotations

import multiprocessing as mp
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import AlgorithmConfig, BaseAlgorithm
from core.layout_trace import trace_layout
from .core import (
    construct_chain_robust_layout,
    normalize_stacks_to_class_ranks,
    robust_exact_badly_placed,
    robust_upper_badly_placed,
    theorem3_has_robust_solution,
    theorem4_unique_priorities_has_robust_solution,
)
from .master_mip import solve_upper_bound_master


class BogeGoerigkKnust2020RobustPMP(BaseAlgorithm):
    name = "Boge-Goerigk-Knust (2020) Robust PMP"
    category = "Exact"
    description = (
        "[single-bay origin] Robust pre-marshalling under uncertain priority-class "
        "order using adjacent-swap uncertainty (EJOR 2020)."
    )
    compatible_problems = ["CRP-Prem"]
    step_label = "Seed"
    requires_solver = True
    solver_backend = "gurobi"

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    def train(
        self,
        problem_factory: Callable,
        result_queue: mp.Queue,
        stop_event: mp.Event,
    ) -> None:
        cfg = self.config
        n_seeds = max(1, int(cfg.num_eval_seeds))
        delta = max(0, int(cfg.extra.get("robust_delta", 2)))
        exact_eval_max_classes = int(cfg.extra.get("exact_eval_max_classes", 9))
        exact_eval_max_scenarios = int(cfg.extra.get("exact_eval_max_scenarios", 100000))
        all_metrics: List[Dict[str, float]] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break
            t0 = time.perf_counter()

            env = problem_factory()
            env.config.seed = seed
            env.reset()

            stack_keys = list(env.yard.stacks.keys())
            key_to_idx = {k: i for i, k in enumerate(stack_keys)}
            max_tiers = int(env.config.max_tiers)

            stacks_raw = {
                key_to_idx[k]: list(st.priority_snapshot())
                for k, st in env.yard.stacks.items()
            }
            stacks, _, _ = normalize_stacks_to_class_ranks(stacks_raw)
            n_stacks = len(stacks)
            item_classes = [cls for arr in stacks.values() for cls in arr]
            n_items = len(item_classes)
            n_classes = len(set(item_classes))
            unique_priorities = n_items == n_classes

            trace_layout(
                f"BogeGoerigkKnust2020 seed={seed+1}/{n_seeds} "
                f"delta={delta} classes={n_classes} items={n_items}"
            )

            # Paper theoretical checks (existence of strict robust layout).
            theorem3_ok = theorem3_has_robust_solution(
                n_items=n_items,
                n_stacks=n_stacks,
                max_tiers=max_tiers,
                delta=delta,
            )
            theorem4_ok = unique_priorities and theorem4_unique_priorities_has_robust_solution(
                n_items=n_items,
                n_stacks=n_stacks,
                max_tiers=max_tiers,
                delta=delta,
            )

            used_theorem = theorem3_ok or theorem4_ok
            if used_theorem:
                target_stacks = construct_chain_robust_layout(
                    item_classes=item_classes,
                    n_stacks=n_stacks,
                    max_tiers=max_tiers,
                    delta=delta,
                )
                upper_obj = robust_upper_badly_placed(target_stacks, delta=delta)
            else:
                master = solve_upper_bound_master(
                    item_classes=item_classes,
                    n_stacks=n_stacks,
                    max_tiers=max_tiers,
                    delta=delta,
                )
                target_stacks = master.target_stacks
                upper_obj = int(master.upper_obj)

            robust_upper = robust_upper_badly_placed(target_stacks, delta=delta)

            robust_exact = None
            if n_classes <= exact_eval_max_classes:
                robust_exact = robust_exact_badly_placed(
                    stacks=target_stacks,
                    n_classes=n_classes,
                    delta=delta,
                    max_scenarios=exact_eval_max_scenarios,
                )

            elapsed = time.perf_counter() - t0
            primary = float(robust_exact if robust_exact is not None else robust_upper)
            metrics = {
                "robust_delta": float(delta),
                "robust_upper_bad_overlaps": float(robust_upper),
                "robust_exact_bad_overlaps": float(-1 if robust_exact is None else robust_exact),
                "upper_mip_obj": float(upper_obj),
                "strict_robust_found": 1.0 if robust_upper == 0 else 0.0,
                "used_theorem_existence": 1.0 if used_theorem else 0.0,
                "unique_priorities": 1.0 if unique_priorities else 0.0,
                "time": float(elapsed),
            }
            all_metrics.append(metrics)

            if primary < self._best_metric:
                self._best_metric = primary
                # This embedding currently optimizes robust layout quality, not move sequence.
                self._best_solution = []

            self._push(
                result_queue,
                step=seed + 1,
                metric=primary,
                metrics=metrics,
                progress=(seed + 1) / n_seeds,
                snapshot=env.get_state_snapshot(),
            )

        if all_metrics:
            agg = {
                k: float(np.mean([m[k] for m in all_metrics if k in m]))
                for k in all_metrics[0]
            }
            self._push(
                result_queue,
                step=n_seeds,
                metric=self._best_metric,
                metrics=agg,
                progress=1.0,
            )

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "num_eval_seeds": {
                "type": "int", "default": 10, "min": 1, "max": 100,
                "label": "Evaluation seeds",
            },
            "robust_delta": {
                "type": "int", "default": 2, "min": 0, "max": 100,
                "label": "Uncertainty budget delta (adjacent swaps)",
            },
            "exact_eval_max_classes": {
                "type": "int", "default": 9, "min": 1, "max": 50,
                "label": "Max classes for exact adversary enumeration",
            },
            "exact_eval_max_scenarios": {
                "type": "int", "default": 100000, "min": 10, "max": 5000000,
                "label": "Scenario cap for exact adversary evaluation",
            },
        })
        return base
