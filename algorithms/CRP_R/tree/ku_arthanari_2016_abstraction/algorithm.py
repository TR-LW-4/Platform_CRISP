"""
Ku and Arthanari abstraction search for CRP-R
<2016> <exact> <restricted> <single-bay> <CRP-R>
time_limit --- 300.0 --- Per-instance wall-clock limit (s)
pdb_depth --- 5 --- PDB remaining-container depth
cache_depth --- 20 --- Forward cache depth
max_cache_size --- 1000000 --- Max forward-cache entries
pdb_build_time_limit --- 60.0 --- PDB build time limit (s)

------------------------------- Reference --------------------------------
D. Ku, T.S. Arthanari,
"On the abstraction method for the container relocation problem",
Computers & Operations Research 68 (2016) 110–122.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Callable, Optional

from web.backend.header_schema import parse_header

from .search import search


class KuArthanari2016Abstraction:

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
        pdb_depth = self._take(header, "pdb_depth", int)
        cache_depth = self._take(header, "cache_depth", int)
        max_cache_size = self._take(header, "max_cache_size", int)
        pdb_build_time_limit = self._take(header, "pdb_build_time_limit", float)
        env = problem_factory()
        stacks = [
            [int(container.priority) for container in stack.containers]
            for _, stack in sorted(env.yard.stacks.items())
        ]
        n_total = int(env.config.num_containers)
        max_tiers = int(env.config.max_tiers)
        report_marks = 50
        reported = -1

        def report(elapsed: float, cost: int) -> None:
            nonlocal reported
            mark = min(int(elapsed / time_limit * report_marks), report_marks - 1)
            if mark == reported:
                return
            reported = mark
            result_queue.put({
                "step": mark + 1,
                "metric": float(cost),
                "progress": min(elapsed / time_limit, 1.0),
            })

        solution, cost = search(
            stacks,
            n_total,
            max_tiers,
            time_limit,
            pdb_depth=pdb_depth,
            cache_depth=cache_depth,
            max_cache_size=max_cache_size,
            pdb_build_time_limit_s=pdb_build_time_limit,
            stop=stop_event.is_set,
            on_progress=report,
        )
        self._best_solution = solution
        result_queue.put({
            "step": report_marks,
            "metric": None if cost is None else float(cost),
            "progress": 1.0,
            "solution": solution,
        })

    def get_best_solution(self) -> Optional[list]:
        return self._best_solution
