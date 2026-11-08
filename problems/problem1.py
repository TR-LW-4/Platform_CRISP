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


