"""
The runtime of a CRP-Time algorithm, measured the same way for every algorithm:
the wall clock of the algorithm itself, from the moment it has its input (the
bay read from the problem) until its final plan exists in its own form.
Progress pushes, the evaluations made only for them, and logging are paused;
building the problem, the replay into a RelocationPlan and finish() are outside.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator, Optional


class Stopwatch:
    """Pausable wall clock; elapsed is the running time in seconds.
    Example:
        timer = Stopwatch()
        timer.start()
        ...search...
        with timer.paused():
            self._push(...)
        timer.stop()
        metrics, env = finish(problem_factory, plan, spec, self.name, timer.elapsed)
    """

    def __init__(self) -> None:
        self.elapsed = 0.0
        self._since: Optional[float] = None

    def start(self) -> None:
        if self._since is None:
            self._since = time.perf_counter()

    def stop(self) -> float:
        if self._since is not None:
            self.elapsed += time.perf_counter() - self._since
            self._since = None
        return self.elapsed

    @contextmanager
    def paused(self) -> Iterator[None]:
        running = self._since is not None
        self.stop()
        try:
            yield
        finally:
            if running:
                self.start()
