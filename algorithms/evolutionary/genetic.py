"""
Genetic Algorithm for container relocation problems.

Chromosome encoding
-------------------
A chromosome is a fixed-length integer vector of length `chrom_len`.
Each gene is an action index (0 … action_space.n − 1).

The GA simulates each chromosome on a fresh problem instance and uses
total primary metric (relocations / shifters / moves) as fitness.

Genetic operators
-----------------
- Selection : tournament selection
- Crossover : uniform crossover (gene-by-gene random choice from parents)
- Mutation  : random gene replacement with probability `mutation_rate`
- Elitism   : top `elite_count` individuals survive unchanged
"""

from __future__ import annotations

import multiprocessing as mp
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig


class GeneticAlgorithm(BaseAlgorithm):

    name        = "Genetic Algorithm"
    category    = "Evolutionary"
    description = ("Population-based genetic algorithm. "
                   "Uniform crossover, random mutation, tournament selection.")
    compatible_problems: List[str] = []   # works with all problems

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)
        cfg = self.config
        self._rng = np.random.RandomState(cfg.seed)

    # ---------------------------------------------------------------- #
    # Train (runs in subprocess)                                         #
    # ---------------------------------------------------------------- #

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        cfg = self.config
        rng = np.random.RandomState(cfg.seed)

        # ── Infer chromosome length from problem ─────────────────── #
        env = problem_factory()
        obs, info = env.reset()
        chrom_len    = cfg.extra.get("chrom_len", cfg.num_steps * 2)
        n_actions    = env.action_space.n
        pop_size     = cfg.population_size
        max_gen      = cfg.extra.get("max_generations", cfg.max_iterations)
        cfg.report_interval = cfg.extra.get("report_every", cfg.report_interval)

        # ── Initialise population ─────────────────────────────────── #
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

            # ── Report progress ───────────────────────────────────── #
            if gen % cfg.report_interval == 0 or gen == max_gen:
                env2 = problem_factory()
                env2.reset()
                snapshot = env2.get_state_snapshot()
                self._push(
                    result_queue,
                    step     = gen,
                    metric   = gen_best_fit,
                    metrics  = {
                        "best_fitness": best_fitness,
                        "mean_fitness": float(np.mean(fitness_scores)),
                        "population_size": pop_size,
                    },
                    progress = gen / max_gen,
                    snapshot = snapshot,
                )

            # ── Next generation ───────────────────────────────────── #
            sorted_idx = np.argsort(fitness_scores).tolist()
            elite      = [population[i][:] for i in sorted_idx[: cfg.elite_count]]

            new_pop: List[List[int]] = list(elite)
            while len(new_pop) < pop_size:
                p1 = self._tournament(population, fitness_scores, cfg.tournament_size, rng)
                p2 = self._tournament(population, fitness_scores, cfg.tournament_size, rng)
                child = self._crossover(p1, p2, rng)
                child = self._mutate(child, n_actions, cfg.mutation_rate, rng)
                new_pop.append(child)
            population = new_pop

    # ---------------------------------------------------------------- #
    # Chromosome evaluation                                              #
    # ---------------------------------------------------------------- #

    def _evaluate(
        self,
        chromosome: List[int],
        problem_factory: Callable,
    ) -> float:
        """Simulate chromosome on a fresh episode; return primary metric."""
        env = problem_factory()
        env.reset()
        total_reward = 0.0
        for action in chromosome:
            _, reward, done, _, _ = env.step(action)
            total_reward += reward
            if done:
                break
        # Primary metric: negative total reward (lower is better)
        return -total_reward

    # ---------------------------------------------------------------- #
    # Genetic operators                                                  #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _tournament(
        population:     List[List[int]],
        fitness:        List[float],
        k:              int,
        rng:            np.random.RandomState,
    ) -> List[int]:
        indices = rng.choice(len(population), size=min(k, len(population)), replace=False)
        best    = indices[int(np.argmin([fitness[i] for i in indices]))]
        return population[best][:]

    @staticmethod
    def _crossover(
        p1:  List[int],
        p2:  List[int],
        rng: np.random.RandomState,
    ) -> List[int]:
        mask  = rng.rand(len(p1)) > 0.5
        child = [p1[i] if mask[i] else p2[i] for i in range(len(p1))]
        return child

    @staticmethod
    def _mutate(
        chromosome:     List[int],
        n_actions:      int,
        mutation_rate:  float,
        rng:            np.random.RandomState,
    ) -> List[int]:
        for i in range(len(chromosome)):
            if rng.rand() < mutation_rate:
                chromosome[i] = int(rng.randint(0, n_actions))
        return chromosome

    # ---------------------------------------------------------------- #
    # Best solution                                                       #
    # ---------------------------------------------------------------- #

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    # ---------------------------------------------------------------- #
    # Config schema                                                       #
    # ---------------------------------------------------------------- #

    step_label = "Generation"

    @classmethod
    def config_schema(cls) -> Dict:
        return {
            "max_generations": {"type": "int",   "default": 200,  "min": 10,    "max": 2000,
                                "label": "Generations (max_generations)",
                                "help": "Total number of evolutionary generations."},
            "report_every":    {"type": "int",   "default": 5,    "min": 1,     "max": 50,
                                "label": "Report every N generations"},
            "population_size": {"type": "int",   "default": 50,   "min": 10,    "max": 500,
                                "label": "Population size"},
            "crossover_rate":  {"type": "float", "default": 0.8,  "min": 0.1,   "max": 1.0,
                                "label": "Crossover rate"},
            "mutation_rate":   {"type": "float", "default": 0.1,  "min": 0.001, "max": 0.5,
                                "label": "Mutation rate"},
            "tournament_size": {"type": "int",   "default": 3,    "min": 2,     "max": 10,
                                "label": "Tournament size"},
            "elite_count":     {"type": "int",   "default": 2,    "min": 0,     "max": 20,
                                "label": "Elite survivors per generation"},
            "chrom_len":       {"type": "int",   "default": 200,  "min": 10,    "max": 2000,
                                "label": "Chromosome length",
                                "help": "Length of action sequence per individual."},
            "seed":            {"type": "int",   "default": 0,    "min": 0,     "max": 9999,
                                "label": "Random seed"},
        }
