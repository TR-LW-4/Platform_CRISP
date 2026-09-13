"""
CRP-Time
<time> <distinct> <CRP-Time>
Fixed-order retrieval minimizing crane working time

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP".
--------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Tuple

import numpy as np

from core.objectives import (
    KinematicsModel,
    ObjectiveSpec,
    evaluate_plan_objectives,
)
from core.plan import Movement, RelocationPlan
from problems.CRP_R import CRP_R


class CRP_Time(CRP_R):
    """Fixed-order retrieval; minimise total crane working time."""

    name         = "CRP-Time"
    description  = (
        "Restricted retrieval with a single crane-time objective. "
        "Relocations are reported; bi-objective search is CRP-MO."
    )
    tags         = ["crp", "time", "fixed-order", "yard-only"]
    metric_names = [
        "objective_value",
        "relocations",
        "crane_time",
        "crane_time_f2",
        "crane_time_vertical",
        "crane_time_rmgc",
        "steps",
    ]

    def _hooks_clear_episode(self) -> None:
        self._plan = RelocationPlan()

    def _hook_after_relocate(
        self,
        src: Tuple[int, int],
        dst: Tuple[int, int],
        container_id: int,
    ) -> None:
        # Hook runs after Yard.relocate(): recover the old source tier and
        # the new destination tier from the resulting stack heights.
        self._plan.add(
            Movement(
                container_id,
                src,
                dst,
                from_tier=self.yard.stacks[src].height + 1,
                to_tier=self.yard.stacks[dst].height,
            )
        )

    def _hook_after_retrieve(self, bay: int, row: int, container_id: int) -> None:
        pos = (bay, row)
        self._plan.add(
            Movement(
                container_id,
                pos,
                None,
                from_tier=self.yard.stacks[pos].height + 1,
            )
        )

    def _objective_metrics(self) -> Dict[str, float]:
        spec = ObjectiveSpec.from_config(self.config)
        if spec.mode != "crane_time":
            spec = replace(spec, mode="crane_time")
        metrics = evaluate_plan_objectives(
            self._plan,
            spec,
            kinematics=KinematicsModel.from_config_extra(self.config.extra),
        )
        metrics["steps"] = float(self._total_steps)
        metrics["progress"] = min(
            1.0,
            self._current_target_priority / max(self.config.num_containers, 1),
        )
        return metrics

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, float]]:
        obs, reward, done, trunc, info = super().step(action)
        if done:
            metrics = self._objective_metrics()
            reward = -float(metrics["objective_value"])
            info.update(metrics)
        elif reward == -1.0:
            reward = 0.0
        return obs, reward, done, trunc, info

    def evaluate(self, solution: List[int]) -> Dict[str, float]:
        saved_seed = self.config.seed
        self.reset()
        for action in solution:
            if self._done:
                break
            self.step(action)
        self.config.seed = saved_seed
        return self._objective_metrics()

    def validate_plan(self, plan: RelocationPlan) -> Dict[str, float]:
        metrics = super().validate_plan(plan)
        if metrics.get("feasible", 0.0):
            objective_metrics = self._objective_metrics()
            metrics.update(objective_metrics)
        else:
            metrics["objective_value"] = float("inf")
        return metrics

    def get_metrics(self) -> Dict[str, float]:
        return self._objective_metrics()

    @classmethod
    def config_schema(cls) -> Dict:
        schema = super().config_schema()
        for key in (
            "gantry_s_per_bay",
            "trolley_s_per_row",
            "gantry_accel_s",
            "spreader_s",
        ):
            schema[key]["visibleWhen"] = {
                "key": "time_model",
                "values": ["rmgc_current"],
            }
        schema.update({
            "objective_mode": {
                "type": "str",
                "default": "crane_time",
                "options": ["crane_time"],
                "label": "Optimization objective",
                "help": (
                    "Single-objective crane working time. Relocations are "
                    "reported but not optimised. Bi-objective search belongs "
                    "on CRP-MO."
                ),
            },
            "time_model": {
                "type": "str",
                "default": "f2",
                "options": ["f2", "f2_vertical", "rmgc_current"],
                "label": "Crane-time model",
                "help": "Voß–Schwarze f2/f2vert, or the legacy multi-bay RMGC model.",
            },
            "stack_s_per_stack": {
                "type": "float", "default": 1.2, "min": 0.0,
                "label": "Horizontal time ts (s/stack)",
                "visibleWhen": {
                    "key": "time_model",
                    "values": ["f2", "f2_vertical"],
                },
            },
            "pickup_place_s": {
                "type": "float", "default": 30.0, "min": 0.0,
                "label": "Fixed pickup/place-down tpp (s)",
                "visibleWhen": {"key": "time_model", "values": ["f2"]},
            },
            "empty_vertical_s_per_tier": {
                "type": "float", "default": 2.59, "min": 0.0,
                "label": "Empty spreader tr0 (s/tier)",
                "visibleWhen": {"key": "time_model", "values": ["f2_vertical"]},
            },
            "loaded_vertical_s_per_tier": {
                "type": "float", "default": 5.18, "min": 0.0,
                "label": "Loaded spreader tr1 (s/tier)",
                "visibleWhen": {"key": "time_model", "values": ["f2_vertical"]},
            },
            "outside_height": {
                "type": "float", "default": 1.5, "min": 0.0,
                "label": "Outside/truck height hout",
                "visibleWhen": {"key": "time_model", "values": ["f2_vertical"]},
            },
        })
        return schema


# legacy alias
CRPTimed = CRP_Time
