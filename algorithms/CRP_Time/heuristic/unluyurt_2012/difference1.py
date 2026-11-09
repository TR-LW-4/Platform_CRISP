"""
UnluyurtAydinDifference1
<2012> <heuristic> <time> <single-bay> <CRP-Time> <CRP-R>
Difference1 fixed-rule relocation heuristic

------------------------------- Reference --------------------------------
T. Ünlüyurt, C. Aydın,
"Improved rehandling strategies for the container retrieval process",
Journal of Advanced Transportation 46 (2012) 378–393.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Callable, Dict, List, Optional

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig

from .retrieve import run
from .rules import difference1


class UnluyurtAydinDifference1(BaseAlgorithm):

    name = "Ünlüyurt–Aydın (2012) Difference1"
    category = "Heuristic"
    description = (
        "Fixed-rule heuristic (Sec. 4.2): each container above the next one to retrieve goes to a stack "
        "where it blocks nothing, preferring the stack whose earliest container is retrieved soonest; "
        "otherwise onto the top container retrieved just before it; otherwise onto the top container "
        "closest to it in retrieval order. It does not use crane time; f2 and f2vert are measured on its "
        "plan. Not in the paper: ties go to the stack nearest the truck lane, and an empty stack counts as "
        "a container retrieved after all others."
    )
    # CRP-R too (Wei, 9 Nov 2026): Difference1 is the paper's heuristic for the relocation count (O9).
    compatible_problems = ["CRP-Time", "CRP-R"]
    geometry = "single-bay"
    objectives = ["fixed-rule"]
    fidelity = "faithful"
    relocation_rule = "restricted"

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        # No parameters: max_iterations and report_interval are not read, and a fixed rule needs no seed.
        for key in ("max_iterations", "report_interval", "seed"):
            base.pop(key)
        return base

    def train(self, problem_factory: Callable, result_queue: mp.Queue, stop_event: mp.Event) -> None:
        run(self, difference1, problem_factory, result_queue, stop_event)

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution
