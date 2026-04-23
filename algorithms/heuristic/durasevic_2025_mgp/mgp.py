"""
Multitask Genetic Programming (MGP) engine — multi-subpopulation evolve
loop + cross-subpop knowledge transfer via crossover and/or migration.

Reference
---------
Ðurasević, M.; Ðumić, M.; Gil-Gala, F. J. (2025)
"Multitask genetic programming for automated design of heuristics for
the container relocation problem",
Engineering Applications of Artificial Intelligence 144: 110001.

Core idea (paper Alg. 4)
------------------------
* Divide the total population into ``SP`` subpopulations, one per task.
* Each subpopulation evolves **independently** via the standard
  steady-state GP loop (3-tournament + subtree crossover + subtree
  mutation), using its OWN fitness function (its "task").
* Knowledge is **shared across subpopulations** by one (or both) of:

    (A) cross-subpop crossover     – when picking parent 2 for a
                                     crossover, with prob ``p_ct`` draw
                                     it from a randomly chosen OTHER
                                     subpop instead of the current one.
    (B) migration                  – every ``migration_frequency``
                                     evaluations, migrate ``M``
                                     individuals from ``P_i`` to
                                     ``P_{(i+1) mod SP}`` in a ring.
                                     Migrants are re-evaluated on the
                                     destination's fitness (so incoming
                                     evaluations also count against the
                                     budget, matching the paper).

Reuse
-----
* ``Node``, ``init_population_rhh``, ``subtree_crossover``,
  ``subtree_mutation`` come directly from
  ``algorithms.heuristic.durasevic_2024.gp_core`` — the 2024 single-task
  engine. Only the **outer control flow** is new here.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from algorithms.heuristic.durasevic_2024.gp_core import (
    Node,
    init_population_rhh,
    subtree_crossover,
    subtree_mutation,
)


# ================================================================ #
#  Task and subpopulation dataclasses                                 #
# ================================================================ #

@dataclass
class Task:
    """
    One 'column' of the multitask problem.

    ``fitness_fn`` takes a Node and returns a scalar cost (lower = better).
    ``name``       is a human-readable tag such as ``"max_tiers=6"`` or
                   ``"objective=crane_time"`` used for progress reporting.
    ``meta``       carries auxiliary info that the orchestrator may need
                   later (e.g., the ``ProblemConfig`` used to build test
                   instances for this task).
    """
    name:       str
    fitness_fn: Callable[[Node], float]
    meta:       Dict = field(default_factory=dict)


@dataclass
class SubPopState:
    """Holds the current state of one subpopulation during evolution."""
    task:          Task
    individuals:   List[Node]
    fitnesses:     List[float]
    best_tree:     Node
    best_fitness:  float


# ================================================================ #
#  Initialisation                                                     #
# ================================================================ #

def _init_subpop(
    task:           Task,
    pop_size:       int,
    max_depth:      int,
    terminal_names: List[str],
    rng:            random.Random,
) -> SubPopState:
    pop  = init_population_rhh(pop_size, max_depth, terminal_names, rng)
    fits = [task.fitness_fn(t) for t in pop]

    best_idx = min(range(len(pop)), key=lambda i: fits[i])
    return SubPopState(
        task         = task,
        individuals  = pop,
        fitnesses    = fits,
        best_tree    = pop[best_idx].copy(),
        best_fitness = fits[best_idx],
    )


# ================================================================ #
#  Migration (ring topology)                                          #
# ================================================================ #

def _migration_step(
    subpops:     List[SubPopState],
    migration_M: int,
    rng:         random.Random,
) -> int:
    """
    Move ``M`` random individuals from P_i into P_{(i+1) mod SP} and
    replace the worst M in the destination.

    Each migrant must be **re-evaluated** on the destination's fitness
    (paper §4.3).  The number of added evaluations is returned so the
    caller can debit them from the remaining budget.
    """
    SP = len(subpops)
    if SP < 2 or migration_M <= 0:
        return 0

    # Snapshot migrants before mutating destinations.
    migrants: List[List[Node]] = []
    for sp in subpops:
        m = min(migration_M, len(sp.individuals))
        idx_sample = rng.sample(range(len(sp.individuals)), m)
        migrants.append([sp.individuals[k].copy() for k in idx_sample])

    added_evals = 0
    for i in range(SP):
        target = subpops[(i + 1) % SP]
        incoming: List[Node] = migrants[i]
        if not incoming:
            continue

        incoming_fits = [target.task.fitness_fn(t) for t in incoming]
        added_evals  += len(incoming_fits)

        # Replace worst M in destination (ties broken arbitrarily).
        worst_sorted = sorted(
            range(len(target.fitnesses)),
            key=lambda k: target.fitnesses[k],
            reverse=True,
        )[: len(incoming)]

        for slot, new_tree, new_fit in zip(
            worst_sorted, incoming, incoming_fits
        ):
            target.individuals[slot] = new_tree
            target.fitnesses[slot]   = new_fit
            if new_fit < target.best_fitness:
                target.best_fitness = new_fit
                target.best_tree    = new_tree.copy()

    return added_evals


# ================================================================ #
#  Cross-subpop parent selection                                      #
# ================================================================ #

def _three_tournament_best(
    sp:  SubPopState,
    rng: random.Random,
) -> int:
    """Return the index of the best among 3 random individuals."""
    idxs = rng.sample(range(len(sp.individuals)), 3)
    return min(idxs, key=lambda k: sp.fitnesses[k])


# ================================================================ #
#  Main MGP driver                                                     #
# ================================================================ #

def evolve_mgp(
    tasks:               List[Task],
    terminal_names:      List[str],
    pop_size_per_subpop: int,
    max_depth:           int,
    max_evals:           int,
    mutation_prob:       float,
    transfer_mode:       str,           # 'crossover' | 'migration' | 'both' | 'none'
    p_ct:                float,
    migration_M:         int,
    migration_frequency: int,
    rng:                 random.Random,
    report_cb:           Optional[Callable[[int, List[SubPopState]], None]] = None,
    stop_flag:           Optional[Callable[[], bool]] = None,
) -> List[SubPopState]:
    """
    Run Multitask GP (paper Algorithm 4) and return the final list of
    subpopulation states.  Each ``SubPopState`` holds its task, current
    population, and best-so-far tree + fitness.

    Notes
    -----
    * ``transfer_mode='none'``  ≡  running ``len(tasks)`` independent
      GP loops with the same random seed (useful as a sanity-check).
    * ``transfer_mode='crossover'`` uses cross-subpop parent selection.
    * ``transfer_mode='migration'`` uses ring-topology migration.
    * ``transfer_mode='both'``      uses both knowledge-transfer paths.
    """
    SP = len(tasks)
    use_crossover = transfer_mode in ("crossover", "both")
    use_migration = transfer_mode in ("migration", "both")

    # ── Initialisation ────────────────────────────────────────── #
    subpops: List[SubPopState] = [
        _init_subpop(t, pop_size_per_subpop, max_depth, terminal_names, rng)
        for t in tasks
    ]
    evals                   = sum(len(sp.individuals) for sp in subpops)
    evals_since_migration   = 0

    if report_cb is not None:
        report_cb(evals, subpops)

    # ── Main loop ────────────────────────────────────────────── #
    while evals < max_evals:
        if stop_flag is not None and stop_flag():
            break

        # One 'round': create one child per subpop (paper Alg 4 inner loop).
        for sp_idx, sp in enumerate(subpops):
            if evals >= max_evals:
                break

            # Select three candidates from THIS subpop.
            tri = rng.sample(range(len(sp.individuals)), 3)
            tri_sorted = sorted(tri, key=lambda k: sp.fitnesses[k])
            p1_idx, p2_idx_same, loser_idx = tri_sorted

            parent1 = sp.individuals[p1_idx]

            # Select parent 2, either from same subpop or (knowledge
            # transfer) from another subpop.
            if use_crossover and SP > 1 and rng.random() < p_ct:
                other_choices = [k for k in range(SP) if k != sp_idx]
                other         = subpops[rng.choice(other_choices)]
                p2_other_idx  = _three_tournament_best(other, rng)
                parent2       = other.individuals[p2_other_idx]
            else:
                parent2 = sp.individuals[p2_idx_same]

            # Genetic operators.
            child = subtree_crossover(parent1, parent2, max_depth, rng)
            if rng.random() < mutation_prob:
                child = subtree_mutation(
                    child, max_depth, terminal_names, rng,
                )

            # Evaluate on THIS subpop's task (paper §4.3 explicit).
            child_fit = sp.task.fitness_fn(child)
            evals              += 1
            evals_since_migration += 1

            # Steady-state replacement (worst of the three).
            sp.individuals[loser_idx] = child
            sp.fitnesses[loser_idx]   = child_fit
            if child_fit < sp.best_fitness:
                sp.best_fitness = child_fit
                sp.best_tree    = child.copy()

        # Migration check (after a full round).
        if (
            use_migration
            and SP > 1
            and evals_since_migration >= migration_frequency
            and evals < max_evals
        ):
            added = _migration_step(subpops, migration_M, rng)
            evals += added
            evals_since_migration = 0

        if report_cb is not None:
            report_cb(evals, subpops)

    return subpops
