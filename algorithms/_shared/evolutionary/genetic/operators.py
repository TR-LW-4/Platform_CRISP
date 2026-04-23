"""
Genetic operators for the GA algorithm.

Separated so they can be unit-tested and reused by derived EA variants
(e.g. memetic algorithms, island models) without touching the training loop.
"""

from __future__ import annotations

from typing import List

import numpy as np


def tournament_select(
    population: List[List[int]],
    fitness:    List[float],
    k:          int,
    rng:        np.random.RandomState,
) -> List[int]:
    """
    Tournament selection: pick k candidates at random, return the fittest.
    """
    indices = rng.choice(len(population), size=min(k, len(population)), replace=False)
    best    = indices[int(np.argmin([fitness[i] for i in indices]))]
    return population[best][:]


def uniform_crossover(
    p1:  List[int],
    p2:  List[int],
    rng: np.random.RandomState,
) -> List[int]:
    """
    Uniform crossover: each gene is independently taken from p1 or p2.
    """
    mask  = rng.rand(len(p1)) > 0.5
    return [p1[i] if mask[i] else p2[i] for i in range(len(p1))]


def random_mutation(
    chromosome:    List[int],
    n_actions:     int,
    mutation_rate: float,
    rng:           np.random.RandomState,
) -> List[int]:
    """
    Random mutation: replace each gene with a random action with probability
    `mutation_rate`.  Mutates in-place and returns the chromosome.
    """
    for i in range(len(chromosome)):
        if rng.rand() < mutation_rate:
            chromosome[i] = int(rng.randint(0, n_actions))
    return chromosome
