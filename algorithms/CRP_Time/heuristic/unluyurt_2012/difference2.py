"""
UnluyurtAydinDifference2
<2012> <heuristic> <time> <single-bay> <CRP-Time>
Difference2 fixed-rule relocation heuristic with a truck-lane preference

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
from .rules import difference2


class UnluyurtAydinDifference2(BaseAlgorithm):

    name = "Ünlüyurt–Aydın (2012) Difference2"
    category = "Heuristic"
    description = (
        "Fixed-rule heuristic (Sec. 4.2): the three rules of Difference1, where at each rule the stacks "
        "between the container's stack and the truck lane are tried before the stacks beyond it. It does "
        "not use the time model's parameters; f2 and f2vert are measured on its plan. Not in the paper: "
        "the lane side is preferred per rule rather than before all rules (the paper allows both "
        "readings), ties go to the stack nearest the truck lane, and an empty stack counts as a container "
        "retrieved after all others."
    )
    compatible_problems = ["CRP-Time"]
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
        run(self, difference2, problem_factory, result_queue, stop_event)

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution
