"""
CRP-MO
<multi-objective> <distinct> <CRP-MO>
Restricted retrieval with the bi-objective (relocations, crane time)

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP".
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from problems.CRP_Time import CRP_Time


class CRP_MO(CRP_Time):
    """
    Same restricted retrieval dynamics as CRP-Time; the search target is the
    pair (relocations, crane_time), not a scalar.

    A single episode still yields one feasible plan. A Pareto set is produced
    by a multi-objective algorithm across many plans; this class only evaluates
    one vector. Set metrics (hypervolume, non-dominated count) stay empty
    until such an algorithm is registered.
    """

    name = "CRP-MO"
    description = (
        "Bi-objective restricted retrieval: minimise relocations and crane "
        "working time together. Returns a vector, not a scalar."
    )
    tags = ["crp", "multi-objective", "fixed-order", "yard-only"]
    metric_names = [
        "relocations",
        "crane_time",
        "crane_time_f2",
        "crane_time_vertical",
        "crane_time_rmgc",
        "steps",
        "n_nondominated",
        "hypervolume",
    ]

    def evaluate_vector(self, solution: List[int]) -> Tuple[float, float]:
        """Return ``(relocations, crane_time)`` for one action sequence."""
        metrics = self.evaluate(solution)
        return float(metrics["relocations"]), float(metrics["crane_time"])

    def _set_placeholders(self, metrics: Dict[str, float]) -> Dict[str, float]:
        metrics.setdefault("n_nondominated", 0.0)
        metrics.setdefault("hypervolume", float("nan"))
        return metrics

    def get_metrics(self) -> Dict[str, float]:
        return self._set_placeholders(super().get_metrics())

    def evaluate(self, solution: List[int]) -> Dict[str, float]:
        return self._set_placeholders(super().evaluate(solution))

    @classmethod
    def config_schema(cls) -> Dict:
        schema = super().config_schema()
        for key in ("objective_mode", "relocation_weight", "time_weight"):
            schema.pop(key, None)
        return schema
