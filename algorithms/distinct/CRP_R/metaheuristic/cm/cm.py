"""
Corridor method for CRP-R
<2011> <metaheuristic> <restricted> <single-bay> <CRP-R>
delta --- 2 --- Stacks kept in the corridor
time_limit --- 5.0 --- CPU time limit in seconds
seed --- 0 --- Random seed
operator --- min_max --- Greedy completion rule

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

import importlib
import time

import numpy as np

from algorithms.ALGORITHM import Algorithm

from .greedy import greedy
from .trajectory import trajectory


class CM(Algorithm):

    def main(self):
        delta = self.parameter("delta", 2, int)
        time_limit = self.parameter("time_limit", 5.0, float)
        rng = np.random.default_rng(self.parameter("seed", 0, int))
        name = self.parameter("operator", "min_max", str)
        module = importlib.import_module("algorithms.distinct.CRP_R.operator." + name)
        choose_destination = module.choose_destination

        stacks = self.problem.stacks
        n_total = self.problem.num_containers
        max_tiers = self.problem.max_tiers
        best, best_cost = greedy(
            stacks, n_total, max_tiers, choose_destination,
        )
        started = time.process_time()

        def expired() -> bool:
            return time.process_time() - started >= time_limit

        while not expired():
            best, best_cost = trajectory(
                stacks, n_total, delta, max_tiers,
                best, best_cost, rng, expired, choose_destination,
            )
        return best