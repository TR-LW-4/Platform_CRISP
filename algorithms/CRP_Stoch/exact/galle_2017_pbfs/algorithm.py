"""
Galle et al. — Pruning-Best-First-Search (PBFS / PBFSA) for the SCRP.

Three algorithms from the same reference implementation, following the
grouping convention used elsewhere in this codebase (e.g.
``zehendner_2017_level/algorithm.py`` bundles L + R + M in one file):

- ``PBFSBatchExact``  : exact batch-model solver (``PBFS`` in the paper),
  run here via ``PBFSA`` with ``error_gap=0``.
- ``PBFSAApprox``     : randomized approximate batch-model solver with a
  bounded average error (``PBFSA``, Section 4 of the paper).
- ``PBFSOnlineExact`` : exact online-model solver (``PBFS_Online``).

All three explore a chance/decision tree with best-first pruning driven
by the Blocking lower bound and a rolling look-ahead lower bound, and
fall back to a deterministic A* tail solver (also ported) once the full
retrieval order of the last batch is known.

Ported from the reference implementation
------------------------------------------
`https://github.com/vgalle/StochasticCRP` (vendored at
`/data/liuw2/StochasticCRP-master`).  Core search logic lives in
:mod:`core.stoch.galle_2017_source`.

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
    blocking_lower_bound,
    env_to_source_bay,
    pbfs_online,
    pbfsa,
)


class _PBFSBase(BaseAlgorithm):
    category = "Exact"
    compatible_problems = ["CRP-Stoch"]
    step_label = "Seed"
    _mode = "batch_exact"

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    def _run_one(self, env, rng: np.random.RandomState) -> Dict[str, float]:
        bay = env_to_source_bay(env, mode="batch")
        t_size, s_size = bay.shape
        n_containers = int(np.sum(bay != 0))
        if n_containers > t_size * s_size - (t_size - 1):
            return {
                "expected_relocations": float("inf"),
                "solve_time": 0.0,
                "nodes_expanded": 0.0,
                "cache_hits": 0.0,
                "num_batches": float(env.num_batches()),
                "num_realizations": float(env.num_realizations()),
            }
        lower_bound_type = int(self.config.extra.get("lower_bound_type", 1))
        time_limit_s = float(self.config.extra.get("time_limit_s", 3600.0))
        if self._mode == "batch_exact":
            obj, stats, elapsed = pbfsa(
                bay,
                lower_bound_type=lower_bound_type,
                error_gap=0.0,
                time_limit_s=time_limit_s,
                rng=rng,
            )
        elif self._mode == "batch_approx":
            error_relative = float(self.config.extra.get("error_relative", 0.5))
            error_gap = float(self.config.extra.get("error_gap", 0.0))
            if error_gap <= 0.0:
                error_gap = max(1.0, error_relative * blocking_lower_bound(bay))
            obj, stats, elapsed = pbfsa(
                bay,
                lower_bound_type=lower_bound_type,
                error_gap=error_gap,
                time_limit_s=time_limit_s,
                rng=rng,
            )
        elif self._mode == "online_exact":
            obj, stats, elapsed = pbfs_online(
                bay,
                lower_bound_type=lower_bound_type,
                time_limit_s=time_limit_s,
                rng=rng,
            )
        else:
            raise ValueError(f"Unknown mode {self._mode}")
        return {
            "expected_relocations": float(obj),
            "solve_time": float(elapsed),
            "nodes_expanded": float(stats.get("nodes_expanded", 0)),
            "cache_hits": float(stats.get("cache_hits", 0)),
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
                    "default": 3,
                    "min": 1,
                    "max": 50,
                    "label": "Evaluation seeds",
                },
                "lower_bound_type": {
                    "type": "int",
                    "default": 1,
                    "min": 0,
                    "max": 3,
                    "label": "Rolling lower-bound depth",
                },
                "time_limit_s": {
                    "type": "float",
                    "default": 3600.0,
                    "min": 1.0,
                    "max": 86400.0,
                    "label": "Time limit (seconds)",
                },
                "error_relative": {
                    "type": "float",
                    "default": 0.5,
                    "min": 0.0,
                    "max": 10.0,
                    "label": "Relative error multiplier",
                    "help": "Used by PBFSAApprox when no absolute error_gap is set.",
                },
                "error_gap": {
                    "type": "float",
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100000.0,
                    "label": "Absolute expected error gap",
                    "help": "0 means derive an absolute gap from error_relative * blocking_lower_bound.",
                },
            }
        )
        return base


class PBFSBatchExact(_PBFSBase):
    name = "Galle et al. PBFS [batch exact]"
    description = (
        "Exact batch-model Pruning-Best-First-Search for the SCRP.  Explores "
        "the chance/decision tree with best-first pruning driven by the "
        "blocking + rolling lower bounds, falling back to an exact A* tail "
        "solver once the last batch's order is fully known."
    )
    _mode = "batch_exact"


class PBFSAApprox(_PBFSBase):
    name = "Galle et al. PBFSA [batch approx]"
    category = "Heuristic"
    description = (
        "Randomized approximate PBFS (PBFSA) for the SCRP with a bounded "
        "average error: samples a controlled number of intra-batch "
        "permutations at each chance node instead of full enumeration, "
        "trading a known error bound for tractability on larger batches."
    )
    _mode = "batch_approx"


class PBFSOnlineExact(_PBFSBase):
    name = "Galle et al. PBFS [online exact]"
    description = (
        "Exact online-model Pruning-Best-First-Search (PBFS_Online): reveals "
        "one container at a time and solves the induced chance/decision tree "
        "exactly, used as the online-model benchmark for E[R]."
    )
    _mode = "online_exact"
