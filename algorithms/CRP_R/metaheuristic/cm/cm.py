"""
Corridor method for CRP-R
<2011> <metaheuristic> <restricted> <single-bay> <CRP-R>
delta --- 2 --- Stacks kept in the corridor
time_limit --- 5.0 --- CPU time limit in seconds

------------------------------- Reference --------------------------------
M. Caserta, S. Voß, M. Sniedovich,
"Applying the corridor method to a blocks relocation problem",
OR Spectrum 33 (2011) 915–929.
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

import numpy as np

from web.backend.header_schema import parse_header

from .greedy import greedy
from .trajectory import trajectory


class CM:

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
        delta = self._take(header, "delta", int)
        time_limit = self._take(header, "time_limit", float)
        seed = getattr(self.config, "seed", 0)
        rng = np.random.default_rng(seed)

        env = problem_factory()
        stacks_init = [
            [int(container.priority) for container in stack.containers]
            for stack in env.yard.stacks.values()
        ]
        n_total = int(env.config.num_containers)
        max_tiers = int(env.config.max_tiers)
        best, best_cost = greedy(stacks_init, n_total, max_tiers)
        started = time.process_time()
        report_marks = 50
        reported = -1

        def expired() -> bool:
            return stop_event.is_set() or time.process_time() - started >= time_limit

        while not expired():
            best, best_cost = trajectory(
                stacks_init, n_total, delta, max_tiers, best, best_cost, rng, expired,
            )
            elapsed = time.process_time() - started
            mark = min(int(elapsed / time_limit * report_marks), report_marks - 1)
            if mark != reported:
                reported = mark
                result_queue.put({
                    "step": mark + 1,
                    "metric": best_cost,
                    "progress": min(elapsed / time_limit, 1.0),
                })

        self._best_solution = best
        result_queue.put({
            "step": report_marks,
            "metric": best_cost,
            "progress": 1.0,
            "solution": best,
        })

    def get_best_solution(self) -> Optional[list]:
        return self._best_solution
