"""
Bounded beam search for CRP-R
<2019> <metaheuristic> <restricted> <single-bay> <CRP-R>
time_limit --- 5.0 --- Time limit in seconds

------------------------------- Reference --------------------------------
T. Bacci, S. Mattia, P. Ventura,
"The bounded beam search algorithm for the block relocation problem",
Computers & Operations Research 103 (2019) 252–264.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import multiprocessing as mp
import time
from typing import Callable, Optional

from web.backend.header_schema import parse_header

from .search import search


class BBS:

    def __init__(self, config=None):
        self.config = config
        self._best_solution = None

    def _take(self, header, name, cast):
        extra = getattr(self.config, "extra", None) or {}
        if name in extra and extra[name] not in (None, ""):
            return cast(extra[name])
        return cast(header[name]["default"])

    def train(
        self,
        problem_factory: Callable,
        result_queue: mp.Queue,
        stop_event: mp.Event,
    ) -> None:
        header = parse_header(__doc__ or "")
        time_limit = self._take(header, "time_limit", float)

        env = problem_factory()
        stacks = [
            [int(container.priority) for container in stack.containers]
            for stack in env.yard.stacks.values()
        ]
        n_total = int(env.config.num_containers)
        max_tiers = int(env.config.max_tiers)
        started = time.perf_counter()
        report_marks = 50
        reported = -1

        def report(elapsed: float, cost: float) -> None:
            nonlocal reported
            mark = min(int(elapsed / time_limit * report_marks), report_marks - 1)
            if mark == reported:
                return
            reported = mark
            result_queue.put({
                "step": mark + 1,
                "metric": cost,
                "progress": min(elapsed / time_limit, 1.0),
            })

        solution, cost = search(
            stacks,
            n_total,
            max_tiers,
            time_limit,
            stop=stop_event.is_set,
            on_progress=report,
        )
        self._best_solution = solution
        result_queue.put({
            "step": report_marks,
            "metric": float(cost),
            "progress": 1.0,
            "solution": solution,
        })

    def get_best_solution(self) -> Optional[list]:
        return self._best_solution
