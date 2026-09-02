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

from typing import Dict, List, Tuple

import numpy as np

from core.objectives import KinematicsModel, compute_crane_time
from core.plan import Movement, RelocationPlan
from problems.CRP_R import CRP_R


class CRP_Time(CRP_R):
    """Fixed-order retrieval; minimise total crane working time."""

    name         = "CRP-Time"
    description  = "Crane-aware container retrieval with a working-time objective."
    tags         = ["crp", "time", "fixed-order", "yard-only"]
    metric_names = ["time", "crane_time", "relocations", "steps"]

    def _hooks_clear_episode(self) -> None:
        self._plan = RelocationPlan()

    def _hook_after_relocate(
        self,
        src: Tuple[int, int],
        dst: Tuple[int, int],
        container_id: int,
    ) -> None:
        self._plan.add(Movement(container_id, src, dst))

    def _hook_after_retrieve(self, bay: int, row: int, container_id: int) -> None:
        self._plan.add(Movement(container_id, (bay, row), None))

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, float]]:
        obs, reward, done, trunc, info = super().step(action)
        if done:
            kin = KinematicsModel.from_config_extra(self.config.extra)
            ct = float(compute_crane_time(self._plan, kin))
            reward = -ct
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
        kin = KinematicsModel.from_config_extra(self.config.extra)
        ct = float(compute_crane_time(self._plan, kin))
        return {
            "time":        ct,
            "crane_time":  ct,
            "relocations": float(self._total_relocations),
            "steps":       float(self._total_steps),
        }

    def get_metrics(self) -> Dict[str, float]:
        kin = KinematicsModel.from_config_extra(self.config.extra)
        ct = (
            float(compute_crane_time(self._plan, kin))
            if self._plan.movements
            else 0.0
        )
        return {
            "time":        ct,
            "crane_time":  ct,
            "relocations": float(self._total_relocations),
            "steps":       float(self._total_steps),
            "progress":    self._current_target_priority / max(self.config.num_containers, 1),
        }


# legacy alias
CRPTimed = CRP_Time
