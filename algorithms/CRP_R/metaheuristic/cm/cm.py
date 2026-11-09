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

import time

import numpy as np

from ...algorithm import Algorithm

from .greedy import greedy
from .trajectory import trajectory


class CM(Algorithm):

    def search(self, problem):
        delta = self.parameter("delta", 2, int)
        time_limit = self.parameter("time_limit", 5.0, float)
        rng = np.random.default_rng(getattr(self.config, "seed", 0))

        stacks = problem.stacks
        best, best_cost = greedy(stacks, problem.n, problem.max_tiers)
        started = time.process_time()

        def expired() -> bool:
            return self.stopped() or time.process_time() - started >= time_limit

        while not expired():
            best, best_cost = trajectory(
                stacks, problem.n, delta, problem.max_tiers,
                best, best_cost, rng, expired,
            )
            elapsed = time.process_time() - started
            self.report(min(elapsed / time_limit, 1.0), best_cost)

        return best, best_cost