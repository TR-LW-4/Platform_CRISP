"""
Zhu2012IDAStarR
<2012> <exact> <restricted> <single-bay> <CRP-R>
Iterative deepening A* with LB3 / PR4
time_limit_s --- 300 --- Per-instance wall-clock limit (s)
lb_mode --- LB3 --- Lower bound: LB1 / LB2 / LB3
probe_mode --- PR4 --- Probe heuristic: PR1–PR4

Reference
---------
W. Zhu, H. Qin, A. Lim, H. Zhang,
"Iterative deepening A* algorithms for the container relocation problem",
IEEE Transactions on Automation Science and Engineering 9 (2012) 710–722.
"""

from __future__ import annotations

from algorithms.common.tree import TreeSolver
from algorithms.common.tree.components import (
    LB1,
    LB2,
    LB3,
    PR1,
    PR2,
    PR3,
    PR4,
    PRPlus,
    PRRollout,
    RestrictedCRPSpace,
    SingleMove,
)
from algorithms.common.tree.strategies import IDAStar


class Zhu2012IDAStarR(TreeSolver):
    """Zhu et al. IDA*-R using the public solver API."""

    name = "Zhu et al. (2012) IDA*-R"
    category = "Exact"
    description = "Zhu et al. (2012) IDA*-R using TreeSolver."
    compatible_problems = ["CRP-R"]
    geometry = "single-bay"
    objectives = ["relocations"]
    fidelity = "faithful"

    _LB_TYPES = {
        LB1.name: LB1,
        LB2.name: LB2,
        LB3.name: LB3,
    }
    _LB_MODES = tuple(_LB_TYPES)
    _PROBE_MODES = (PR1, PR2, PR3, PR4)

    def build_search(self) -> IDAStar:
        """Compose the paper's IDA*-R configuration."""
        # Header order: time_limit_s, lb_mode, probe_mode.
        _, lb_mode, probe_mode = self.parameter_set("300", "LB3", "PR4")
        lb_mode = str(lb_mode)
        probe_mode = str(probe_mode)

        if lb_mode not in self._LB_MODES:
            lb_mode = LB3.name
        if probe_mode not in self._PROBE_MODES:
            probe_mode = PR4

        space = RestrictedCRPSpace()
        return IDAStar(
            state_space=space,
            branching=SingleMove(space),
            lower_bound=self._LB_TYPES[lb_mode](space),
            upper_bound=PRPlus(space),
            frontier_probe=PRRollout(space, probe_mode),
        )
