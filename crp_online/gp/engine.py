import random
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
from ..core.instance import ProblemInstance
from ..core.environment import Environment
from ..evaluator.prefix_evaluator import PrefixExpression
from ..heuristics.restricted import solve_restricted
from ..heuristics.unrestricted import solve_unrestricted
from ..config.defaults import (
    GP_POPULATION_SIZE,
    GP_MAX_TREE_DEPTH,
    GP_TOURNAMENT_SIZE,
    GP_MUTATION_RATE,
    GP_CROSSOVER_RATE,
    GP_TOTAL_EVALUATIONS,
    CORE_TERMINALS,
)
from .tree import GPNode
from .initialization import ramped_half_and_half
from .operators import apply_crossover, apply_mutation


@dataclass
class Individual:
    """A GP candidate solution individual with its expression tree and scalar fitness."""
    tree: GPNode
    fitness: float = float("inf")
    compiled_expr: Optional[PrefixExpression] = None

    def get_expression(self) -> PrefixExpression:
        if self.compiled_expr is None:
            self.compiled_expr = PrefixExpression(self.tree.to_prefix_str())
        return self.compiled_expr


@dataclass
class GPEvaluationLog:
    """Record of GP progress at regular evaluation intervals."""
    eval_count: int
    best_fitness: float
    best_expression: str
    elapsed_seconds: float


class SteadyStateGP:
    """Pure-Python Steady-State Tournament Genetic Programming engine.
    
    Implements Algorithm 3 from Đurasević et al. (2026):
    - Tournament selection with replacement of the worst individual in tournament (k=3).
    - Ramped half-and-half initialization.
    - 5 Crossover and 6 Mutation operators.
    - Fixed evaluation budget (default 50,000 evaluations).
    """

    def __init__(
        self,
        pop_size: int = GP_POPULATION_SIZE,
        max_evals: int = GP_TOTAL_EVALUATIONS,
        tournament_size: int = GP_TOURNAMENT_SIZE,
        crossover_rate: float = GP_CROSSOVER_RATE,
        mutation_rate: float = GP_MUTATION_RATE,
        max_tree_depth: int = GP_MAX_TREE_DEPTH,
        functions: Sequence[str] = ("+", "-", "*", "/"),
        terminals: Sequence[str] = CORE_TERMINALS,
        scheme: str = "restricted",
        seed: Optional[int] = None,
    ) -> None:
        self.pop_size: int = pop_size
        self.max_evals: int = max_evals
        self.tournament_size: int = tournament_size
        self.crossover_rate: float = crossover_rate
        self.mutation_rate: float = mutation_rate
        self.max_tree_depth: int = max_tree_depth
        self.functions: Sequence[str] = functions
        self.terminals: Sequence[str] = terminals
        self.scheme: str = scheme
        self.rng: random.Random = random.Random(seed)

    def evaluate_individual(
        self,
        ind: Individual,
        instances: Sequence[ProblemInstance],
        look_ahead_level: float = 1.0,
        horizon_strategy: str = "ignore",
        use_container_groups: bool = False,
        num_groups: int = 2,
        known_within_group: bool = False,
        mode: str = "cpp_compat",
    ) -> float:
        """Evaluates total relocations across the given training instances."""
        expr = ind.get_expression()
        total_relocations = 0

        for inst in instances:
            env = Environment(inst, store_moves=False)
            if self.scheme == "unrestricted":
                moves = solve_unrestricted(
                    env,
                    expr,
                    look_ahead_level=look_ahead_level,
                    horizon_strategy=horizon_strategy,
                    use_container_groups=use_container_groups,
                    num_groups=num_groups,
                    known_within_group=known_within_group,
                    mode=mode,
                )
            else:
                moves = solve_restricted(
                    env,
                    expr,
                    look_ahead_level=look_ahead_level,
                    horizon_strategy=horizon_strategy,
                    use_container_groups=use_container_groups,
                    num_groups=num_groups,
                    known_within_group=known_within_group,
                    mode=mode,
                )
            total_relocations += env.total_relocations

        ind.fitness = float(total_relocations)
        return ind.fitness

    def train(
        self,
        training_instances: Sequence[ProblemInstance],
        look_ahead_level: float = 1.0,
        horizon_strategy: str = "ignore",
        use_container_groups: bool = False,
        num_groups: int = 2,
        known_within_group: bool = False,
        mode: str = "cpp_compat",
        log_interval: int = 1000,
    ) -> Tuple[Individual, List[GPEvaluationLog]]:
        """Executes the steady-state GP evolutionary loop until max_evals is reached."""
        start_time = time.time()
        logs: List[GPEvaluationLog] = []

        # 1. Initialize population via Ramped Half-and-Half
        raw_trees = ramped_half_and_half(
            self.pop_size,
            min_depth=1,
            max_depth=self.max_tree_depth - 1,
            functions=self.functions,
            terminals=self.terminals,
            rng=self.rng,
        )
        population: List[Individual] = [Individual(tree=t) for t in raw_trees]

        # 2. Evaluate initial population
        eval_count = 0
        best_ind: Optional[Individual] = None

        for ind in population:
            fit = self.evaluate_individual(
                ind,
                training_instances,
                look_ahead_level=look_ahead_level,
                horizon_strategy=horizon_strategy,
                use_container_groups=use_container_groups,
                num_groups=num_groups,
                known_within_group=known_within_group,
                mode=mode,
            )
            eval_count += 1
            if best_ind is None or fit < best_ind.fitness:
                best_ind = Individual(tree=ind.tree.clone(), fitness=fit)

        logs.append(
            GPEvaluationLog(
                eval_count=eval_count,
                best_fitness=best_ind.fitness,
                best_expression=best_ind.tree.to_prefix_str(),
                elapsed_seconds=time.time() - start_time,
            )
        )

        # 3. Steady-State Tournament Loop
        while eval_count < self.max_evals:
            # Tournament selection (sample k individuals without replacement)
            tournament_indices = self.rng.sample(range(self.pop_size), self.tournament_size)
            # Sort tournament participants by fitness (lower is better)
            tournament_indices.sort(key=lambda idx: population[idx].fitness)

            # Two best are parents
            p1 = population[tournament_indices[0]]
            p2 = population[tournament_indices[1]]
            # Worst in tournament is target for replacement
            worst_idx = tournament_indices[-1]

            # Crossover
            if self.rng.random() < self.crossover_rate:
                child_tree = apply_crossover(p1.tree, p2.tree, self.max_tree_depth, self.rng)
            else:
                child_tree = p1.tree.clone()

            # Mutation
            if self.rng.random() < self.mutation_rate:
                child_tree = apply_mutation(
                    child_tree,
                    self.functions,
                    self.terminals,
                    self.max_tree_depth,
                    self.rng,
                )

            # Evaluate offspring
            child = Individual(tree=child_tree)
            fit = self.evaluate_individual(
                child,
                training_instances,
                look_ahead_level=look_ahead_level,
                horizon_strategy=horizon_strategy,
                use_container_groups=use_container_groups,
                num_groups=num_groups,
                known_within_group=known_within_group,
                mode=mode,
            )
            eval_count += 1

            if fit < best_ind.fitness:
                best_ind = Individual(tree=child.tree.clone(), fitness=fit)

            # Replace worst in tournament
            population[worst_idx] = child

            if eval_count % log_interval == 0 or eval_count >= self.max_evals:
                logs.append(
                    GPEvaluationLog(
                        eval_count=eval_count,
                        best_fitness=best_ind.fitness,
                        best_expression=best_ind.tree.to_prefix_str(),
                        elapsed_seconds=time.time() - start_time,
                    )
                )

        assert best_ind is not None
        return best_ind, logs
