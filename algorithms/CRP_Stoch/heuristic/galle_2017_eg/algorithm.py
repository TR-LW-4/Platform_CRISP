"""
ExpectedGroupAssignment
<2018> <heuristic> <stochastic> <single-bay> <CRP-Stoch>
Expected Group Assignment for batch-stochastic relocation
revelation_model --- batch --- Revelation: batch or online
num_samples --- 5000 --- Monte-Carlo realizations for E[R]

------------------------------- Reference --------------------------------
V. Galle, S. Borjian Boroujeni, V.H. Manshadi, C. Barnhart, P. Jaillet,
"The Stochastic Container Relocation Problem",
Transportation Science 52 (2018) 1030–1047.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import AlgorithmConfig, BaseAlgorithm
from core.stoch.galle_2017_source import (
    HEURISTIC_IDS,
    env_to_source_bay,
    run_heuristic_batch,
    run_heuristic_online,
)


class ExpectedGroupAssignment(BaseAlgorithm):

    name = "Galle et al. Expected Group (EG)"
    category = "Heuristic"
    description = (
        "Expected Group Assignment (EG): two-phase, group-aware relocation "
        "heuristic for the SCRP.  Generalises Wu & Ting's (2012) GAH to the "
        "stochastic batch/online setting by assigning several same-batch "
        "blockers to shared destination columns.  Ported from the Galle et "
        "al. SCRP reference implementation."
    )
    compatible_problems = ["CRP-Stoch"]
    geometry = "single-bay"
    objectives = ["relocations"]
    fidelity = "faithful"
    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    def _run_one(self, env, rng: np.random.RandomState) -> Dict[str, float]:
        mode = str(self.config.extra.get("revelation_model", "batch"))
        n_samples = int(self.config.extra.get("num_samples", 5000))
        bay = env_to_source_bay(env, mode=mode)
        t_size, s_size = bay.shape
        n_containers = int(np.sum(bay != 0))
        if n_containers > t_size * s_size - (t_size - 1):
            return {
                "expected_relocations": float("inf"),
                "num_samples": float(n_samples),
                "num_batches": float(env.num_batches()),
                "num_realizations": float(env.num_realizations()),
            }
        heuristic_id = HEURISTIC_IDS["EG"]
        if mode == "online":
            expected = run_heuristic_online(bay, heuristic_id, n_samples, rng=rng)
        else:
            expected = run_heuristic_batch(bay, heuristic_id, n_samples, rng=rng)
        return {
            "expected_relocations": float(expected),
            "num_samples": float(n_samples),
            "num_batches": float(env.num_batches()),
            "num_realizations": float(env.num_realizations()),
        }

    def train(
        self,
        problem_factory: Callable,
        result_queue: mp.Queue,
        stop_event: mp.Event,
    ) -> None:
        cfg = self.config
        n_seeds = 1  # multi-seed eval removed; single run only
        all_metrics: List[Dict[str, float]] = []
        for seed in range(n_seeds):
            if stop_event.is_set():
                break
            env = problem_factory()
            env.reset(options={"skip_auto_retrieve": True})
            rng = np.random.RandomState(seed)
            metrics = self._run_one(env, rng)
            primary = float(metrics["expected_relocations"])
            all_metrics.append(metrics)
            if primary < self._best_metric:
                self._best_metric = primary
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
        base.update(
            {
                "num_samples": {
                    "type": "int",
                    "default": 5000,
                    "min": 1,
                    "max": 100000,
                    "label": "Monte-Carlo samples",
                    "help": "Number of sampled retrieval orders as in the source implementation.",
                },
                "revelation_model": {
                    "type": "str",
                    "default": "batch",
                    "options": ["batch", "online"],
                    "label": "Revelation model",
                },
            }
        )
        return base
