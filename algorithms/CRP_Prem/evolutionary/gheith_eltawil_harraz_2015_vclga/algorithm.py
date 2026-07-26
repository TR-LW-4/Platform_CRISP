"""
Gheith, Eltawil, Harraz (2015) variable-length GA for pre-marshalling.

Reference:
M. Gheith, A.B. Eltawil, N.A. Harraz,
"Solving the container pre-marshalling problem using variable length
genetic algorithms", Engineering Optimization, 2015.
"""

from __future__ import annotations

import multiprocessing as mp
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from core.base_algorithm import AlgorithmConfig, BaseAlgorithm
from .core import (
    Move,
    Stacks,
    EvalResult,
    all_possible_moves,
    cyclic_movements_elimination,
    feasible_solution_implementation,
    growth_mutation,
    minimum_chromosome_length_preservation,
    parent_generation,
    replace_mutation,
    shrink_mutation,
    single_point_crossover,
    swap_mutation,
    total_bad_overlaps,
    tournament_selection,
)


def _encode_action(src_idx: int, dst_idx: int, n_stacks: int) -> int:
    dst_flat = dst_idx if dst_idx < src_idx else dst_idx - 1
    return src_idx * (n_stacks - 1) + dst_flat


class GheithEltawilHarraz2015VCLGA(BaseAlgorithm):
    name = "Gheith–Eltawil–Harraz (2015) VCLGA"
    category = "Evolutionary"
    description = (
        "[single-bay origin] Variable-length genetic algorithm for pre-marshalling. "
        "Single-point crossover with growth/shrink/swap/replace mutations and "
        "chromosome feasibility repair."
    )
    compatible_problems = ["CRP-Prem"]
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
        n_seeds = max(1, int(cfg.num_eval_seeds))
        all_seed_metrics: List[Dict[str, float]] = []

        pop_size = int(cfg.extra.get("population_size", cfg.population_size))
        max_gen = int(cfg.extra.get("max_generations", cfg.max_iterations))
        report_every = int(cfg.extra.get("report_every", cfg.report_interval))
        tournament_k = int(cfg.extra.get("tournament_size", cfg.tournament_size))
        elite_count = int(cfg.extra.get("elite_count", cfg.elite_count))
        crossover_prob = float(cfg.extra.get("crossover_probability", 0.15))
        p_growth = float(cfg.extra.get("growth_mutation_probability", 0.30))
        p_shrink = float(cfg.extra.get("shrink_mutation_probability", 0.30))
        p_swap = float(cfg.extra.get("swap_mutation_probability", 0.30))
        p_replace = float(cfg.extra.get("replace_mutation_probability", 0.30))
        max_len_factor = float(cfg.extra.get("max_length_factor", 5.0))

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            rng = random.Random(int(cfg.seed) + seed)
            env = problem_factory()
            env.config.seed = seed
            env.reset()

            stack_keys = list(env.yard.stacks.keys())
            n_stacks = len(stack_keys)
            key_to_idx = {k: i for i, k in enumerate(stack_keys)}
            max_tiers = int(env.config.max_tiers)
            n_containers = int(env.config.num_containers)
            move_pool = all_possible_moves(n_stacks)

            stacks0: Stacks = {
                key_to_idx[k]: list(st.priority_snapshot())
                for k, st in env.yard.stacks.items()
            }

            min_len = max(1, total_bad_overlaps(stacks0))
            max_len = max(min_len, int(max_len_factor * max(1, n_containers)))

            population = parent_generation(
                pop_size=pop_size,
                move_pool=move_pool,
                min_len=min_len,
                max_len=max_len,
                rng=rng,
            )

            best_eval: Optional[EvalResult] = None

            for gen in range(1, max_gen + 1):
                if stop_event.is_set():
                    break

                evaluated: List[EvalResult] = []
                for chrom in population:
                    repaired = cyclic_movements_elimination(chrom)
                    repaired = minimum_chromosome_length_preservation(
                        repaired, min_len=min_len, move_pool=move_pool, rng=rng
                    )
                    ev = feasible_solution_implementation(
                        stacks_init=stacks0,
                        chromosome=repaired,
                        max_tiers=max_tiers,
                    )
                    evaluated.append(ev)

                evaluated.sort(key=lambda e: (e.bad_overlaps, e.moves_used, e.fitness))
                if best_eval is None or (evaluated[0].bad_overlaps, evaluated[0].moves_used) < (best_eval.bad_overlaps, best_eval.moves_used):
                    best_eval = evaluated[0]
                    actions = [_encode_action(s, d, n_stacks) for (s, d) in best_eval.executed]
                    self._best_solution = actions
                    self._best_metric = float(best_eval.fitness)

                if gen % max(1, report_every) == 0 or gen == max_gen:
                    self._push(
                        result_queue,
                        step=gen + seed * max_gen,
                        metric=float(evaluated[0].fitness),
                        metrics={
                            "best_bad_overlaps": float(evaluated[0].bad_overlaps),
                            "best_moves": float(evaluated[0].moves_used),
                            "mean_bad_overlaps": float(np.mean([e.bad_overlaps for e in evaluated])),
                            "mean_moves": float(np.mean([e.moves_used for e in evaluated])),
                            "time": float(evaluated[0].moves_used),
                        },
                        progress=(gen + seed * max_gen) / max(1, n_seeds * max_gen),
                        snapshot=env.get_state_snapshot(),
                    )

                # Generate next population.
                next_pop: List[List[Move]] = [e.chromosome[:] for e in evaluated[: max(0, elite_count)]]
                while len(next_pop) < pop_size:
                    p1 = tournament_selection(evaluated, tournament_k, rng).chromosome
                    p2 = tournament_selection(evaluated, tournament_k, rng).chromosome

                    if rng.random() <= crossover_prob:
                        c1, c2 = single_point_crossover(p1, p2, rng)
                    else:
                        c1, c2 = list(p1), list(p2)

                    # Mutation family from the paper's variable-length setting.
                    if rng.random() <= p_growth:
                        c1 = growth_mutation(c1, move_pool, rng)
                    if rng.random() <= p_shrink:
                        c1 = shrink_mutation(c1, rng)
                    if rng.random() <= p_swap:
                        c1 = swap_mutation(c1, rng)
                    if rng.random() <= p_replace:
                        c1 = replace_mutation(c1, move_pool, rng)

                    c1 = cyclic_movements_elimination(c1)
                    c1 = minimum_chromosome_length_preservation(c1, min_len, move_pool, rng)
                    if len(c1) > max_len:
                        c1 = c1[:max_len]
                    next_pop.append(c1)

                    if len(next_pop) < pop_size:
                        if rng.random() <= p_growth:
                            c2 = growth_mutation(c2, move_pool, rng)
                        if rng.random() <= p_shrink:
                            c2 = shrink_mutation(c2, rng)
                        if rng.random() <= p_swap:
                            c2 = swap_mutation(c2, rng)
                        if rng.random() <= p_replace:
                            c2 = replace_mutation(c2, move_pool, rng)
                        c2 = cyclic_movements_elimination(c2)
                        c2 = minimum_chromosome_length_preservation(c2, min_len, move_pool, rng)
                        if len(c2) > max_len:
                            c2 = c2[:max_len]
                        next_pop.append(c2)

                population = next_pop

            if best_eval is None:
                best_eval = feasible_solution_implementation(stacks0, [], max_tiers)
            seed_metrics = {
                "bad_overlaps": float(best_eval.bad_overlaps),
                "moves": float(best_eval.moves_used),
                "time": float(best_eval.moves_used),
                "solved": 1.0 if best_eval.bad_overlaps == 0 else 0.0,
            }
            all_seed_metrics.append(seed_metrics)
            self._push(
                result_queue,
                step=seed + 1,
                metric=float(best_eval.fitness),
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
            "num_eval_seeds": {
                "type": "int", "default": 10, "min": 1, "max": 100,
                "label": "Evaluation seeds",
            },
            "population_size": {
                "type": "int", "default": 50, "min": 10, "max": 500,
                "label": "Population size",
            },
            "max_generations": {
                "type": "int", "default": 300, "min": 10, "max": 100000,
                "label": "Maximum generations",
            },
            "report_every": {
                "type": "int", "default": 10, "min": 1, "max": 500,
                "label": "Report every N generations",
            },
            "tournament_size": {
                "type": "int", "default": 3, "min": 2, "max": 20,
                "label": "Tournament size",
            },
            "elite_count": {
                "type": "int", "default": 2, "min": 0, "max": 100,
                "label": "Elite survivors per generation",
            },
            "crossover_probability": {
                "type": "float", "default": 0.15, "min": 0.0, "max": 1.0,
                "label": "Single-point crossover probability",
            },
            "growth_mutation_probability": {
                "type": "float", "default": 0.30, "min": 0.0, "max": 1.0,
                "label": "Growth mutation probability",
            },
            "shrink_mutation_probability": {
                "type": "float", "default": 0.30, "min": 0.0, "max": 1.0,
                "label": "Shrink mutation probability",
            },
            "swap_mutation_probability": {
                "type": "float", "default": 0.30, "min": 0.0, "max": 1.0,
                "label": "Swap mutation probability",
            },
            "replace_mutation_probability": {
                "type": "float", "default": 0.30, "min": 0.0, "max": 1.0,
                "label": "Replace mutation probability",
            },
            "max_length_factor": {
                "type": "float", "default": 5.0, "min": 1.0, "max": 50.0,
                "label": "Maximum chromosome length factor",
            },
        })
        return base

