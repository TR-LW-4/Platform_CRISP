"""
PeteringHussein2013BRPIII
<2013> <exact> <unrestricted> <single-bay> <CRP-U>
BRP-III mixed-integer program with cleaning moves
time_limit_s --- 3600 --- Gurobi time limit (s)

------------------------------- Reference --------------------------------
M.E.H. Petering, M.I. Hussein,
"A new mixed integer program and extended look-ahead heuristic algorithm
 for the block relocation problem",
European Journal of Operational Research 231 (2013) 120–130.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import math
import multiprocessing as mp
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

from core.base_algorithm import AlgorithmConfig, BaseAlgorithm
from core.plan import Movement, RelocationPlan

from .brp_iii import solve_brp_iii


class PeteringHussein2013BRPIII(BaseAlgorithm):
    """BRP-III time-indexed MIP for the unrestricted BRP."""

    name = "Petering & Hussein (2013) BRP-III [Gurobi]"
    category = "Exact"
    requires_solver = True
    solver_backend = "gurobi"
    description = (
        "[CRP-U] Petering & Hussein (EJOR 2013) BRP-III mixed-integer "
        "formulation. Minimises total moves, permits cleaning moves, and uses "
        "LA-1 to set the move horizon. [Requires Gurobi license]"
    )
    compatible_problems = ["CRP-U"]
    geometry = "single-bay"
    objectives = ["relocations"]
    fidelity = "faithful"
    def __init__(self, config: Optional[AlgorithmConfig] = None) -> None:
        super().__init__(config)

    def train(
        self,
        problem_factory: Callable,
        result_queue: mp.Queue,
        stop_event: mp.Event,
    ) -> None:
        cfg = self.config
        extra = cfg.extra or {}
        n_seeds = 1  # multi-seed eval removed; single run only
        time_limit_s = float(extra.get("time_limit_s", 3600.0))
        output_flag = int(extra.get("output_flag", 0))
        enforce_safety = bool(extra.get("enforce_adjacent_height", False))
        verbose = bool(extra.get("verbose", False))
        all_metrics: List[Dict] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.reset(seed=seed, options={"skip_auto_retrieve": True})

            initial_yard = env.yard
            stack_keys = sorted(initial_yard.stacks.keys())
            stacks = [
                [container.priority for container in initial_yard.stacks[key].containers]
                for key in stack_keys
            ]
            C = len(env.containers)
            S = len(stack_keys)
            H = env.config.max_tiers

            # Section 3.2 of the paper sets W to the number of moves in the
            # feasible LA-1 solution, avoiding unnecessary variables.
            try:
                horizon = _la1_move_upper_bound(stacks, H)
            except RuntimeError as exc:
                self._report_error(
                    result_queue,
                    seed,
                    n_seeds,
                    f"LA-1 failed to produce a horizon upper bound: {exc}",
                )
                continue

            try:
                import gurobipy  # noqa: F401
            except ImportError:
                self._report_error(
                    result_queue,
                    seed,
                    n_seeds,
                    "gurobipy is not installed",
                )
                continue

            started = time.perf_counter()
            try:
                result = solve_brp_iii(
                    stacks=stacks,
                    max_height=H,
                    horizon=horizon,
                    time_limit_s=time_limit_s,
                    output_flag=output_flag,
                    enforce_adjacent_height=enforce_safety,
                )
            except Exception as exc:
                self._report_error(
                    result_queue,
                    seed,
                    n_seeds,
                    f"Gurobi/BRP-III failed: {exc}",
                )
                continue
            elapsed = time.perf_counter() - started

            raw_moves = result.get("moves")
            if raw_moves is None:
                metrics = {
                    "relocations": float("inf"),
                    "steps": float("inf"),
                    "time": float("inf"),
                    "total_moves": float("inf"),
                    "feasible": 0.0,
                    "optimal_proven": 0.0,
                    "time_out": 1.0 if result.get("time_out") else 0.0,
                    "horizon_W": float(horizon),
                    "n_vars": float(result.get("n_vars", 0)),
                    "n_constrs": float(result.get("n_constrs", 0)),
                    "solve_time_s": round(elapsed, 4),
                }
                all_metrics.append(metrics)
                self._push(
                    result_queue,
                    step=seed + 1,
                    metric=float("inf"),
                    metrics=metrics,
                    progress=(seed + 1) / n_seeds,
                    snapshot=env.get_state_snapshot(),
                    extra={"seed": seed, "error": result.get("error")},
                )
                continue

            plan = _moves_to_plan(raw_moves, stack_keys, env.containers)
            plan_metrics = env.evaluate_plan(plan)
            feasible = (
                bool(plan_metrics.get("feasible", 0.0))
                and _validate_model_moves(raw_moves, stacks, H)
            )
            n_relocations = int(result["relocations"])
            action_list = _plan_to_actions(plan, stack_keys)
            primary = float(n_relocations) if feasible else float("inf")

            metrics = {
                "relocations": float(n_relocations),
                "steps": float(result["total_moves"]),
                "time": float(n_relocations),
                "total_moves": float(result["total_moves"]),
                "feasible": 1.0 if feasible else 0.0,
                "optimal_proven": 1.0 if result["optimal"] else 0.0,
                "time_out": 1.0 if result["time_out"] else 0.0,
                "objective_last_retrieval": float(result["obj"]),
                "horizon_W": float(horizon),
                "n_vars": float(result["n_vars"]),
                "n_constrs": float(result["n_constrs"]),
                "solve_time_s": round(elapsed, 4),
            }
            all_metrics.append(metrics)

            if primary < self._best_metric:
                self._best_metric = primary
                self._best_solution = action_list

            self._push(
                result_queue,
                step=seed + 1,
                metric=primary,
                metrics=metrics,
                progress=(seed + 1) / n_seeds,
                snapshot=env.get_state_snapshot(),
                extra={
                    "seed": seed,
                    "horizon_W": horizon,
                    "total_moves": result["total_moves"],
                    "relocations": n_relocations,
                    "optimal": result["optimal"],
                },
            )

            if verbose:
                print(
                    f"[PeteringHussein2013BRPIII] seed={seed} C={C} S={S} "
                    f"H={H} W={horizon} reloc={n_relocations} "
                    f"moves={result['total_moves']} optimal={result['optimal']} "
                    f"feasible={feasible} t={elapsed:.2f}s",
                    file=sys.stderr,
                    flush=True,
                )

        if all_metrics:
            successful_metrics = [
                metrics
                for metrics in all_metrics
                if (
                    math.isfinite(metrics.get("relocations", float("inf")))
                    and metrics.get("feasible", 0.0) == 1.0
                )
            ]
            if successful_metrics:
                aggregate = {
                    key: (
                        sum(
                            float(metrics[key])
                            for metrics in successful_metrics
                            if key in metrics
                        )
                        / sum(1 for metrics in successful_metrics if key in metrics)
                    )
                    for key in successful_metrics[0]
                }
                self._push(
                    result_queue,
                    step=n_seeds,
                    metric=self._best_metric,
                    metrics=aggregate,
                    progress=1.0,
                )

    def _report_error(
        self,
        result_queue: mp.Queue,
        seed: int,
        n_seeds: int,
        message: str,
    ) -> None:
        print(
            f"[PeteringHussein2013BRPIII] ERROR: {message}",
            file=sys.stderr,
            flush=True,
        )
        self._push(
            result_queue,
            step=seed + 1,
            metric=float("inf"),
            metrics={
                "relocations": float("inf"),
                "feasible": 0.0,
                "error": 1.0,
            },
            progress=(seed + 1) / n_seeds,
            extra={"seed": seed, "error_message": message},
        )

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "time_limit_s": {
                "type": "float",
                "default": 3600.0,
                "min": 1.0,
                "max": 86400.0,
                "label": "Time limit (s)",
            },
            "output_flag": {
                "type": "int",
                "default": 0,
                "min": 0,
                "max": 1,
                "label": "Gurobi output",
                "help": "0 = silent; 1 = print the Gurobi solve log.",
            },
            "enforce_adjacent_height": {
                "type": "bool",
                "default": False,
                "label": "Adjacent-stack safety",
                "help": (
                    "Enable BRP-III constraints (7)-(8), limiting adjacent "
                    "stack height differences to two. Disabled in the paper's "
                    "BRP-I versus BRP-III experiments."
                ),
            },
            "verbose": {
                "type": "bool",
                "default": False,
                "label": "Verbose logging",
            },
        })
        return base


def _moves_to_plan(
    raw_moves: List[Tuple[int, int, Optional[int]]],
    stack_keys: List[Tuple[int, int]],
    containers: list,
) -> RelocationPlan:
    """Convert model priorities/stack indices to platform plan movements."""
    id_by_priority = {container.priority: container.id for container in containers}
    plan = RelocationPlan()
    for priority, src, dst in raw_moves:
        plan.add(Movement(
            container_id=id_by_priority[priority],
            from_pos=stack_keys[src],
            to_pos=None if dst is None else stack_keys[dst],
        ))
    return plan


def _validate_model_moves(
    raw_moves: List[Tuple[int, int, Optional[int]]],
    initial_stacks: List[List[int]],
    max_height: int,
) -> bool:
    """Validate LIFO access, retrieval order, capacity, and final emptiness."""
    sim = [list(stack) for stack in initial_stacks]
    next_target = 1

    for priority, src, dst in raw_moves:
        if src < 0 or src >= len(sim) or not sim[src]:
            return False
        if sim[src][-1] != priority:
            return False

        sim[src].pop()
        if dst is None:
            if priority != next_target:
                return False
            next_target += 1
            continue

        if dst < 0 or dst >= len(sim) or dst == src:
            return False
        if len(sim[dst]) >= max_height:
            return False
        sim[dst].append(priority)

    C = sum(len(stack) for stack in initial_stacks)
    return next_target == C + 1 and all(not stack for stack in sim)


def _la1_move_upper_bound(stacks: List[List[int]], max_height: int) -> int:
    """Return a feasible total-move upper bound using bounded-height LA-1."""
    sim = [list(stack) for stack in stacks]
    C = sum(len(stack) for stack in sim)
    moves = 0

    def low(stack_index: int) -> float:
        return float(min(sim[stack_index])) if sim[stack_index] else float("inf")

    for target in range(1, C + 1):
        source = next(
            (index for index, stack in enumerate(sim) if target in stack),
            None,
        )
        if source is None:
            raise RuntimeError(f"target priority {target} is missing")

        while sim[source][-1] != target:
            blocker = sim[source].pop()
            destinations = [
                index
                for index, stack in enumerate(sim)
                if index != source and len(stack) < max_height
            ]
            if not destinations:
                raise RuntimeError("all destination stacks are full")

            good = [
                index
                for index in destinations
                if sim[index] and low(index) > blocker
            ]
            destination = (
                min(good, key=low)
                if good
                else max(destinations, key=low)
            )
            sim[destination].append(blocker)
            moves += 1

        sim[source].pop()
        moves += 1

    return moves


def _plan_to_actions(
    plan: RelocationPlan,
    stack_keys: List[Tuple[int, int]],
) -> List[int]:
    """Convert relocation movements to CRP-U's flat src*S+dst encoding."""
    stack_index = {key: index for index, key in enumerate(stack_keys)}
    S = len(stack_keys)
    return [
        stack_index[move.from_pos] * S + stack_index[move.to_pos]
        for move in plan.movements
        if not move.is_retrieval
    ]
