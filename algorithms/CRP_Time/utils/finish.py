"""
The last step of every CRP-Time algorithm: the problem validates the final plan
and reports its metrics, so all algorithms report the same metrics, computed by
the platform from a plan that obeys the problem's rules.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from core.objectives import ObjectiveSpec, objective_value
from core.plan import RelocationPlan


def finish(problem_factory: Callable, plan: RelocationPlan, spec: ObjectiveSpec,
           algorithm: str, runtime_s: float) -> Tuple[Dict[str, float], Any]:
    """Metrics of the final plan, and the problem that validated it.

    A new problem replays the plan under its own rules (only the top container of
    the current target's stack moves, retrievals in priority order, no stack above
    its height, an empty bay at the end) and evaluates it; an infeasible plan
    raises. objective_value is the algorithm's own objective under spec, steps the
    number of moves (relocations and retrievals), runtime_s the algorithm's own
    running time (utils.timer).
    Example:
        metrics, env = finish(problem_factory, plan, spec, self.name, timer.elapsed)
    """
    env = problem_factory()
    if not env.validate_plan(plan).get("feasible"):
        raise RuntimeError(f"{algorithm}: plan rejected by validate_plan: "
                           + "; ".join(env.get_last_validation_errors()))
    metrics = env.get_metrics()
    metrics["objective_value"] = objective_value(metrics, spec)
    metrics["steps"] = float(plan.num_moves())
    metrics["runtime_s"] = float(runtime_s)
    return metrics, env
