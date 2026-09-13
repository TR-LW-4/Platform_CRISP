"""
HottungTierney2016BRKGA
<2016> <evolutionary> <premarshalling> <single-bay> <CRP-Prem>
Biased random-key GA with construction decoder
population_size --- 80 --- BRKGA population size
elite_fraction --- 0.20 --- Elite share of the population
mutants_fraction --- 0.20 --- Mutant share of the population
elite_bias --- 0.70 --- Elite-allele inheritance probability
max_generations --- 250 --- Generations per layout

------------------------------- Reference --------------------------------
A. Hottung, K. Tierney,
"A biased random-key genetic algorithm for the container pre-marshalling
 problem",
Computers & Operations Research 75 (2016) 83–102.
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
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.base_algorithm import AlgorithmConfig, BaseAlgorithm
from .core import (
    Move,
    Stacks,
    apply_move,
    clone_stacks,
    decode_brkga_solution,
    total_bad_overlaps,
)
from .vendor_backend import run_vendor_solver


def _encode_action(src_idx: int, dst_idx: int, n_stacks: int) -> int:
    dst_flat = dst_idx if dst_idx < src_idx else dst_idx - 1
    return src_idx * (n_stacks - 1) + dst_flat


def _biased_uniform_crossover(
    elite: Sequence[float],
    non_elite: Sequence[float],
    bias: float,
    rng: random.Random,
) -> List[float]:
    out: List[float] = []
    for i in range(len(elite)):
        if rng.random() < bias:
            out.append(float(elite[i]))
        else:
            out.append(float(non_elite[i]))
    return out


class HottungTierney2016BRKGA(BaseAlgorithm):
    name = "Hottung–Tierney (2016) BRKGA"
    category = "Evolutionary"
    description = (
        "[single-bay origin] Biased random-key GA with construction decoder "
        "for container pre-marshalling."
    )
    compatible_problems = ["CRP-Prem"]
    geometry = "single-bay"
    objectives = ["relocations"]
    fidelity = "faithful"
    step_label = "Generation"

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    def train(
        self,
        problem_factory: Callable,
        result_queue: mp.Queue,
        stop_event: mp.Event,
    ) -> None:
        cfg = self.config
        n_seeds = 1  # multi-seed eval removed; single run only
        all_seed_metrics: List[Dict[str, float]] = []

        pop_size = int(cfg.extra.get("population_size", 80))
        elite_frac = float(cfg.extra.get("elite_fraction", 0.20))
        mutants_frac = float(cfg.extra.get("mutants_fraction", 0.20))
        bias = float(cfg.extra.get("elite_bias", 0.70))
        max_generations = int(cfg.extra.get("max_generations", 250))
        report_every = int(cfg.extra.get("report_every", 10))
        max_steps = int(cfg.extra.get("decoder_max_steps", 2000))
        clear_retries = int(cfg.extra.get("decoder_clear_retries", 100))
        chromosome_len = int(cfg.extra.get("chromosome_len", 32))
        backend = str(cfg.extra.get("backend", "auto")).strip().lower()
        vendor_enabled = backend in {"vendor", "auto"}

        for seed in range(n_seeds):
            if stop_event.is_set():
                break
            rng = random.Random(int(cfg.seed) + seed)

            env = problem_factory()
            env.reset()

            stack_keys = list(env.yard.stacks.keys())
            n_stacks = len(stack_keys)
            key_to_idx = {k: i for i, k in enumerate(stack_keys)}
            max_tiers = int(env.config.max_tiers)

            stacks0: Stacks = {
                key_to_idx[k]: list(st.priority_snapshot())
                for k, st in env.yard.stacks.items()
            }

            if vendor_enabled:
                try:
                    vendor = run_vendor_solver(
                        stacks_init=stacks0,
                        max_tiers=max_tiers,
                        seed=int(cfg.seed) + seed,
                        extra=cfg.extra,
                    )
                    best_moves = list(vendor.moves)
                    final_stacks = clone_stacks(stacks0)
                    for mv in best_moves:
                        apply_move(final_stacks, mv, max_tiers=max_tiers)
                    best_bad = total_bad_overlaps(final_stacks)
                    best_fit = float(best_bad * 1_000_000 + len(best_moves))
                    self._best_metric = best_fit
                    self._best_solution = [
                        _encode_action(s, d, n_stacks) for (s, d) in best_moves
                    ]

                    seed_metrics = {
                        "bad_overlaps": float(best_bad),
                        "moves": float(len(best_moves)),
                        "time": float(vendor.duration_seconds),
                        "solved": 1.0 if best_bad == 0 else 0.0,
                        "backend_vendor": 1.0,
                    }
                    all_seed_metrics.append(seed_metrics)
                    self._push(
                        result_queue,
                        step=seed + 1,
                        metric=best_fit,
                        metrics=seed_metrics,
                        progress=(seed + 1) / n_seeds,
                        snapshot=env.get_state_snapshot(),
                    )
                    continue
                except Exception:
                    if backend == "vendor":
                        raise
                    vendor_enabled = False

            population: List[List[float]] = [
                [rng.random() for _ in range(chromosome_len)]
                for _ in range(pop_size)
            ]

            best_moves: List[Move] = []
            best_bad = 10**9
            best_fit = float("inf")

            elite_count = max(1, int(elite_frac * pop_size))
            mutants_count = max(1, int(mutants_frac * pop_size))
            offspring_count = max(0, pop_size - elite_count - mutants_count)

            for gen in range(1, max_generations + 1):
                if stop_event.is_set():
                    break

                scored: List[Tuple[float, int, int, List[float], List[Move]]] = []
                for chrom in population:
                    dec = decode_brkga_solution(
                        stacks_init=stacks0,
                        max_tiers=max_tiers,
                        chromosome=chrom,
                        max_steps=max_steps,
                        clear_retries=clear_retries,
                        rng=rng,
                    )
                    scored.append((dec.fitness, dec.bad_overlaps, len(dec.moves), chrom, dec.moves))
                scored.sort(key=lambda x: (x[0], x[1], x[2]))

                top_fit, top_bad, top_len, _, top_moves = scored[0]
                if (top_bad, top_len) < (best_bad, len(best_moves)):
                    best_bad = top_bad
                    best_moves = list(top_moves)
                    best_fit = top_fit
                    self._best_metric = float(best_fit)
                    self._best_solution = [_encode_action(s, d, n_stacks) for (s, d) in best_moves]

                if gen % max(1, report_every) == 0 or gen == max_generations:
                    self._push(
                        result_queue,
                        step=seed * max_generations + gen,
                        metric=float(top_fit),
                        metrics={
                            "bad_overlaps": float(top_bad),
                            "moves": float(top_len),
                            "best_bad_overlaps": float(best_bad),
                            "best_moves": float(len(best_moves)),
                            "mean_bad_overlaps": float(np.mean([x[1] for x in scored])),
                            "mean_moves": float(np.mean([x[2] for x in scored])),
                            "time": float(top_len),
                        },
                        progress=(seed * max_generations + gen) / max(1, n_seeds * max_generations),
                        snapshot=env.get_state_snapshot(),
                    )

                elite = [scored[i][3] for i in range(elite_count)]
                non_elite = [scored[i][3] for i in range(elite_count, len(scored))]
                if not non_elite:
                    non_elite = elite

                new_pop: List[List[float]] = [list(c) for c in elite]
                for _ in range(offspring_count):
                    e = elite[rng.randrange(len(elite))]
                    n = non_elite[rng.randrange(len(non_elite))]
                    child = _biased_uniform_crossover(e, n, bias=bias, rng=rng)
                    new_pop.append(child)
                for _ in range(mutants_count):
                    new_pop.append([rng.random() for _ in range(chromosome_len)])
                population = new_pop[:pop_size]

            seed_metrics = {
                "bad_overlaps": float(best_bad),
                "moves": float(len(best_moves)),
                "time": float(len(best_moves)),
                "solved": 1.0 if best_bad == 0 else 0.0,
                "backend_vendor": 0.0,
            }
            all_seed_metrics.append(seed_metrics)
            self._push(
                result_queue,
                step=seed + 1,
                metric=float(best_fit),
                metrics=seed_metrics,
                progress=(seed + 1) / n_seeds,
                snapshot=env.get_state_snapshot(),
            )

        if all_seed_metrics:
            agg = {
                k: float(np.mean([m[k] for m in all_seed_metrics if k in m]))
                for k in all_seed_metrics[0]
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
            "backend": {
                "type": "str", "default": "auto",
                "label": "Backend (auto/python/vendor)",
            },
            "population_size": {
                "type": "int", "default": 80, "min": 10, "max": 2000,
                "label": "Population size",
            },
            "max_generations": {
                "type": "int", "default": 250, "min": 10, "max": 10000,
                "label": "Max generations",
            },
            "report_every": {
                "type": "int", "default": 10, "min": 1, "max": 1000,
                "label": "Report every N generations",
            },
            "elite_fraction": {
                "type": "float", "default": 0.20, "min": 0.01, "max": 0.90,
                "label": "Elite fraction",
            },
            "mutants_fraction": {
                "type": "float", "default": 0.20, "min": 0.01, "max": 0.90,
                "label": "Mutants fraction",
            },
            "elite_bias": {
                "type": "float", "default": 0.70, "min": 0.0, "max": 1.0,
                "label": "Biased crossover elite probability",
            },
            "chromosome_len": {
                "type": "int", "default": 32, "min": 8, "max": 1024,
                "label": "Chromosome length",
            },
            "decoder_max_steps": {
                "type": "int", "default": 2000, "min": 10, "max": 100000,
                "label": "Decoder max move steps",
            },
            "decoder_clear_retries": {
                "type": "int", "default": 100, "min": 1, "max": 5000,
                "label": "Clear-stack retries",
            },
            "vendor_biased_percentage": {
                "type": "int", "default": 65, "min": 0, "max": 100,
                "label": "Vendor biased percentage (x)",
            },
            "vendor_max_cpu_time": {
                "type": "int", "default": 1200, "min": 1, "max": 100000,
                "label": "Vendor max CPU time seconds (y)",
            },
            "vendor_min_share_valid": {
                "type": "float", "default": 0.210087, "min": 0.0, "max": 1.0,
                "label": "Vendor min share valid population (z)",
            },
            "max_generations_without_improvement": {
                "type": "int", "default": 74, "min": 1, "max": 100000,
                "label": "Vendor max generations without improvement (w)",
            },
        })
        return base

