"""Parent of every CRP-R algorithm."""

from __future__ import annotations

import inspect
from typing import Callable, Optional

import multiprocessing as mp

from web.backend.header_schema import parse_header


class Bay:
    def __init__(self, stacks, n, max_tiers):
        self.stacks = stacks
        self.n = n
        self.max_tiers = max_tiers


class Algorithm:

    def __init__(self, config=None):
        self.config = config
        self._best_solution = None
        self.result_queue = None
        self.stop_event = None
        self._reported = -1
        self.report_marks = 50

    def parameter(self, name, default, cast):
        header = parse_header(inspect.getmodule(self.__class__).__doc__ or "")
        extra = getattr(self.config, "extra", None) or {}
        if name in extra and extra[name] not in (None, ""):
            return cast(extra[name])
        if name in header:
            return cast(header[name]["default"])
        return default

    def search(self, problem):
        raise NotImplementedError

    def solve(self, problem):
        solution, cost = self.search(problem)
        self._best_solution = solution
        return solution, cost

    def train(self, problem_factory: Callable, result_queue: mp.Queue, stop_event: mp.Event) -> None:
        self.result_queue = result_queue
        self.stop_event = stop_event
        self._reported = -1
        solution, cost = self.search(self._as_problem(problem_factory()))
        self._best_solution = solution
        result_queue.put({
            "step": self.report_marks,
            "metric": None if cost is None else float(cost),
            "progress": 1.0,
            "solution": solution,
        })

    def report(self, progress, metric) -> None:
        if self.result_queue is None:
            return
        mark = min(int(float(progress) * self.report_marks), self.report_marks - 1)
        if mark == self._reported:
            return
        self._reported = mark
        self.result_queue.put({
            "step": mark + 1,
            "metric": float(metric),
            "progress": float(progress),
        })

    def stopped(self) -> bool:
        return self.stop_event is not None and self.stop_event.is_set()

    def get_best_solution(self) -> Optional[list]:
        return self._best_solution

    def _as_problem(self, env):
        if all(hasattr(env, name) for name in ("stacks", "n", "max_tiers")):
            return env
        stacks = [
            [int(container.priority) for container in stack.containers]
            for _, stack in sorted(env.yard.stacks.items())
        ]
        return Bay(stacks, int(env.config.num_containers), int(env.config.max_tiers))