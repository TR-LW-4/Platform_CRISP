"""
CRP-R
<restricted> <distinct> <CRP-R>
Classical Block Relocation Problem with fixed retrieval order

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP".
--------------------------------------------------------------------------
"""

from __future__ import annotations

from numbers import Integral
from typing import Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym

from core.base_problem import BaseProblem, ProblemConfig
from core.container import Container
from core.yard import Move, Yard
from core.plan import Movement, RelocationPlan
from core.objectives import KinematicsModel, compute_crane_time, lower_bound_relocations
from core.layout_trace import trace_layout
from core.benchmark_keys import layout_path_from_extra


class CRP_R(BaseProblem):

    name         = "CRP-R"
    description = (
        "Restricted container relocation with fixed retrieval order; "
        "only blockers above the current target may be moved."
    )
    tags         = ["restrict", "fixed-order"]
    metric_names = ["relocations"]

    def __init__(
        self,
        config: Optional[ProblemConfig] = None,
    ):
        super().__init__(config)
        self._current_target_priority: int = 1
        self._total_relocations: int       = 0
        self._total_steps: int             = 0
        self._done: bool                   = False
        self._last_validation_errors: List[str] = []



    def reset(
        self,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        
        
        super().reset(options=options)

        self.yard.clear()
        self._total_relocations = 0
        self._total_steps       = 0
        self._done              = False

        self._build_episode()
        opts = options if options is not None else {}
        if opts.get("skip_auto_retrieve"):
            self._current_target_priority = 1
            obs  = self._get_obs()
            info = self._get_info()
            return obs, info
        return self._finish_reset_after_layout_loaded()

    def _finish_reset_after_layout_loaded(self) -> Tuple[np.ndarray, Dict]:
        """Set the retrieval cursor and take any targets already on stack tops."""
        self._current_target_priority = 1
        self._advance_auto_retrievals()

        obs  = self._get_obs()
        info = self._get_info()
        return obs, info

    def _build_episode(self) -> None:
        """Fill yard from ``extra['layout_file_path']`` (Caserta/Zhu-style file)."""
        cfg = self.config
        path_str = layout_path_from_extra(cfg.extra)
        if not path_str:
            raise ValueError(
                f"{self.name} requires a Caserta/Zhu layout file "
                "(set extra['layout_file_path'])."
            )

        from pathlib import Path

        from core.benchmarks.caserta import apply_caserta_file_to_yard

        trace_layout(f"CRP_R._build_episode: read layout file {path_str}")
        self.containers = apply_caserta_file_to_yard(
            self.yard, cfg, Path(path_str)
        )

    # ---------------------------------------------------------------- #
    # Auto-retrieve: pull targets that are already on top               #
    # ------------------- --------------------------------------------- #

    def _advance_auto_retrievals(self) -> None:
        """Retrieve all consecutive targets that are already accessible."""
        while True:
            target = self._get_target_container()
            if target is None:
                self._done = True
                return
            stack = self.yard._find_stack(target)
            if stack is None:
                self._done = True
                return
            if stack.top == target:
                stack.pop()
                self.yard.move_history.append(
                    Move("retrieve", target.id, (stack.bay, stack.row), None)
                )
                self.yard.total_retrievals += 1
                self._current_target_priority += 1
            else:
                return   # blocked — wait for the next relocation