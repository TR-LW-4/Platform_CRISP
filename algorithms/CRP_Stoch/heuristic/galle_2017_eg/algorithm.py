"""
Galle et al. — Expected Group Assignment (EG) heuristic for the SCRP.

EG generalises the deterministic Group Assignment Heuristic of Wu & Ting
(2012) to the stochastic setting: blocking containers above the current
target are relocated in two phases (an "acceptable" descending-label
phase, then an ascending-label phase for the remainder), using virtual
minimum-label bookkeeping so several blockers of the same batch can be
grouped into the same destination column.

Ported from the reference implementation
------------------------------------------
`https://github.com/vgalle/StochasticCRP` (vendored at
`/data/liuw2/StochasticCRP-master`, files `retrieveEG.m` + `heuristic.m`).
Core logic lives in :mod:`core.stoch.galle_2017_source` and is shared
with the ERI heuristic (Ku & Arthanari 2016) and the PBFS/PBFSA exact
algorithms (Galle et al.), which are ported from the same repository.

Reference
---------
V. Galle, S. Borjian Boroujeni, V. H. Manshadi, C. Barnhart, P. Jaillet,
"The Stochastic Container Relocation Problem", 2017 (source repository);
published as Galle et al., *Transportation Science* 52(5), 2018.
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
    step_label = "Seed"

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
        n_seeds = max(1, cfg.num_eval_seeds)
        all_metrics: List[Dict[str, float]] = []
        for seed in range(n_seeds):
            if stop_event.is_set():
                break
            env = problem_factory()
            env.config.seed = seed
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
                "num_eval_seeds": {
                    "type": "int",
                    "default": 5,
                    "min": 1,
                    "max": 200,
                    "label": "Evaluation seeds",
                },
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
