"""
Ku & Arthanari (2016) — Expected Reshuffling Index (ERI) heuristic.

ERI extends Murty et al.'s (2005) deterministic Reshuffle Index (RI) to
the case where several containers share the same departure time window
(the *retrieval order within a window is unknown*): the column score
becomes the *expected* number of containers that depart earlier than the
one being relocated, using the closed-form ``f_n = n/2`` derived in the
paper (Section 4).  The blocking container is relocated to the column
with the lowest ERI (ties broken toward the taller, then leftmost,
column).

This is the paper's main practical contribution — the exact SDP +
abstraction search from the same paper is intentionally not embedded
here (see ``algorithms/CRP_Stoch/README.md`` for the rationale); the
Galle et al. PBFS/PBFSA exact tree search (``exact/galle_2017_pbfs/``)
provides an equivalent exact benchmark for the same problem class.

Ported from the reference implementation
------------------------------------------
`https://github.com/vgalle/StochasticCRP` (vendored at
`/data/liuw2/StochasticCRP-master`, files `retrieveERI.m` + `heuristic.m`
/ `heuristic_Online.m`).  Core logic lives in
:mod:`core.stoch.galle_2017_source`.

Reference
---------
D. Ku, T. S. Arthanari, "Container relocation problem with time windows
for container departure", *European Journal of Operational Research*
252 (2016) 1031-1039.  DOI: 10.1016/j.ejor.2016.01.055
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


class ExpectedReshufflingIndex(BaseAlgorithm):

    name = "Ku & Arthanari (2016) ERI"
    category = "Heuristic"
    description = (
        "Expected Reshuffling Index (ERI): extends the deterministic "
        "Reshuffle Index (Murty et al. 2005) to time-window uncertainty "
        "via the closed-form expected score f_n = n/2 for same-window "
        "containers.  The paper's main practical heuristic for the CRP "
        "with departure time windows (CRPTW)."
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
        heuristic_id = HEURISTIC_IDS["ERI"]
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
                    "help": (
                        "'online' matches the original Ku & Arthanari (2016) setting "
                        "(one container revealed at a time); 'batch' matches the "
                        "Galle et al. SCRP batch model."
                    ),
                },
            }
        )
        return base
