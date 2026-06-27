"""
Genetic Algorithm for container relocation problems.

Chromosome encoding
-------------------
A chromosome is a fixed-length integer vector.
Each gene is an action index (0 … action_space.n − 1).

Genetic operators are defined in operators.py.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from .operators import tournament_select, uniform_crossover, random_mutation


class GeneticAlgorithm(BaseAlgorithm):

    name        = "Genetic Algorithm"
    category    = "Evolutionary"
    description = ("[general]  "
                   "Population-based genetic algorithm. "
                   "Uniform crossover, random mutation, tournament selection.")
    compatible_problems = [
        "CRP-Time",
        "CRP-Prem",
        "CRP-Stow",
        "CRP-Stoch",
        "CRP-U",
        "CRP-D",
    ]

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        cfg = self.config
        rng = np.random.RandomState(cfg.seed)

        env = problem_factory()
        env.reset()
        chrom_len = cfg.extra.get("chrom_len", cfg.num_steps * 2)
        n_actions = env.action_space.n
        pop_size  = cfg.population_size
        max_gen   = cfg.extra.get("max_generations", cfg.max_iterations)
        cfg.report_interval = cfg.extra.get("report_every", cfg.report_interval)

        population = [
            rng.randint(0, n_actions, size=chrom_len).tolist()
            for _ in range(pop_size)
        ]

        best_chromosome: Optional[List[int]] = None
        best_fitness    = float("inf")

        for gen in range(1, max_gen + 1):
            if stop_event.is_set():
                break

            # ── Evaluate ─────────────────────────────────────────── #
            fitness_scores = [
                self._evaluate(chrom, problem_factory)
                for chrom in population
            ]

            # ── Track best ───────────────────────────────────────── #
            gen_best_idx = int(np.argmin(fitness_scores))
            gen_best_fit = fitness_scores[gen_best_idx]
            if gen_best_fit < best_fitness:
                best_fitness    = gen_best_fit
                best_chromosome = population[gen_best_idx][:]
                self._best_solution = best_chromosome

            # ── Report ───────────────────────────────────────────── #
            if gen % cfg.report_interval == 0 or gen == max_gen:
                env2 = problem_factory()
                env2.reset()
                self._push(
                    result_queue,
                    step     = gen,
                    metric   = gen_best_fit,
                    metrics  = {
                        "best_fitness":   best_fitness,
                        "mean_fitness":   float(np.mean(fitness_scores)),
                        "population_size": pop_size,
                    },
                    progress = gen / max_gen,
                    snapshot = env2.get_state_snapshot(),
                )

            # ── Next generation ───────────────────────────────────── #
            sorted_idx = np.argsort(fitness_scores).tolist()
            elite      = [population[i][:] for i in sorted_idx[: cfg.elite_count]]

            new_pop: List[List[int]] = list(elite)
            while len(new_pop) < pop_size:
                p1    = tournament_select(population, fitness_scores, cfg.tournament_size, rng)
                p2    = tournament_select(population, fitness_scores, cfg.tournament_size, rng)
                child = uniform_crossover(p1, p2, rng)
                child = random_mutation(child, n_actions, cfg.mutation_rate, rng)
                new_pop.append(child)
            population = new_pop

    # ---------------------------------------------------------------- #
    # Chromosome evaluation                                              #
    # ---------------------------------------------------------------- #

    def _evaluate(self, chromosome: List[int], problem_factory: Callable) -> float:
        """Simulate chromosome on a fresh episode; return primary metric."""
        env = problem_factory()
        env.reset()
        total_reward = 0.0
        for action in chromosome:
            _, reward, done, _, _ = env.step(action)
            total_reward += reward
            if done:
                break
        return -total_reward

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    step_label = "Generation"

    @classmethod
    def config_schema(cls) -> Dict:
        return {
            "max_generations": {"type": "int",   "default": 200,  "min": 10,    "max": 2000,
                                "label": "Generations"},
            "report_every":    {"type": "int",   "default": 5,    "min": 1,     "max": 50,
                                "label": "Report every N generations"},
            "population_size": {"type": "int",   "default": 50,   "min": 10,    "max": 500,
                                "label": "Population size"},
            "mutation_rate":   {"type": "float", "default": 0.1,  "min": 0.001, "max": 0.5,
                                "label": "Mutation rate"},
            "tournament_size": {"type": "int",   "default": 3,    "min": 2,     "max": 10,
                                "label": "Tournament size"},
            "elite_count":     {"type": "int",   "default": 2,    "min": 0,     "max": 20,
                                "label": "Elite survivors per generation"},
            "chrom_len":       {"type": "int",   "default": 200,  "min": 10,    "max": 2000,
                                "label": "Chromosome length"},
            "seed":            {"type": "int",   "default": 0,    "min": 0,     "max": 9999,
                                "label": "Random seed"},
        }
