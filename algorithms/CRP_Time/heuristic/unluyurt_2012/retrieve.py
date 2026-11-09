"""
The run of both Difference heuristics (Ünlüyurt & Aydın 2012, Sec. 4.2): every
container is retrieved in priority order, and each container above it is
relocated to the stack a selector from rules.py chooses.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section of difference1.py.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from algorithms.CRP_Time.utils.bay import read_bay, target_stack
from algorithms.CRP_Time.utils.crane_time import ObjectiveSpec, check_objective
from algorithms.CRP_Time.utils.finish import finish
from algorithms.CRP_Time.utils.plan import actions, replay, saved_moves
from algorithms.CRP_Time.utils.timer import Stopwatch
from core.base_algorithm import BaseAlgorithm
from core.plan import RelocationPlan

Selector = Callable[[int, int, Sequence[Sequence[int]], int, int], Optional[int]]


def retrieve(stacks: Sequence[Sequence[int]], max_tiers: int, select: Selector) -> List[Tuple[int, Optional[int]]]:
    """Moves (src, dst) on stack indices that empty the bay, dst None for a retrieval.

    Containers 1..N are retrieved in order. While the next one is covered, the top container x
    of its stack is relocated to select(x, src, stacks, max_tiers, N + 1) (restricted, p. 380);
    containers on top at the start are retrieved as moves of the plan. Raises if no other stack
    has room (O6).
    """
    st = [list(s) for s in stacks]
    empty = sum(len(s) for s in st) + 1
    moves: List[Tuple[int, Optional[int]]] = []
    for c in range(1, empty):
        src = target_stack(st, c)
        while st[src][-1] != c:
            x = st[src][-1]
            dst = select(x, src, st, max_tiers, empty)
            if dst is None:
                raise RuntimeError(f"no stack can receive container {x}: every stack other than "
                                   f"stack {src + 1} is full (max_tiers = {max_tiers})")
            st[dst].append(st[src].pop())
            moves.append((src, dst))
        st[src].pop()
        moves.append((src, None))
    return moves


def finish_relocations(problem_factory: Callable, plan: RelocationPlan, algorithm: str,
                       runtime_s: float) -> Tuple[Dict[str, float], Any]:
    """CRP-R counterpart of utils.finish: the metrics of the final plan, and the problem that validated it.

    The metrics are the dict of the problem's validate_plan (relocations, steps = all moves,
    time = the platform's RMGC crane time), not get_metrics(), which reports steps and time
    as the number of relocations under CRP-R. An infeasible plan raises.
    """
    env = problem_factory()
    metrics = env.validate_plan(plan)
    if not metrics.get("feasible"):
        raise RuntimeError(f"{algorithm}: plan rejected by validate_plan: "
                           + "; ".join(env.get_last_validation_errors()))
    metrics["runtime_s"] = float(runtime_s)
    return metrics, env


def run(algorithm: BaseAlgorithm, select: Selector, problem_factory: Callable,
        result_queue: mp.Queue, stop_event: mp.Event) -> None:
    """train() of both heuristics: read the bay, build the plan with select, and report the
    metrics of the plan as validated by the problem. Under CRP-Time (finish) the metric is the
    selected crane time; under CRP-R (Difference1 only, O9; finish_relocations) it is the
    number of relocations."""
    if stop_event.is_set():
        return
    env = problem_factory()
    stacks, positions = read_bay(env)
    relocations_only = env.name == "CRP-R"
    if not relocations_only:
        spec = ObjectiveSpec.from_config(env.config)
        check_objective(spec, algorithm.name)

    timer = Stopwatch()
    timer.start()
    moves = retrieve(stacks, int(env.config.max_tiers), select)
    timer.stop()

    plan = replay(stacks, positions, moves)
    if relocations_only:
        metrics, env_final = finish_relocations(problem_factory, plan, algorithm.name, timer.elapsed)
        metric = float(metrics["relocations"])
    else:
        metrics, env_final = finish(problem_factory, plan, spec, algorithm.name, timer.elapsed)
        metric = float(metrics["objective_value"])
    algorithm._best_solution = actions(plan, env.config.num_rows)
    algorithm._push(result_queue, step=1, metric=metric, metrics=metrics,
                    progress=1.0, snapshot=env_final.get_state_snapshot(), extra={"moves": saved_moves(plan)})
