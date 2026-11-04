"""
ExpositoBB2015
<2015> <exact> <restricted> <single-bay> <CRP-R>
Best-first branch-and-bound with alpha-restricted branching
alpha --- 0 --- Max children per node (0 = unlimited / optimal)
time_limit_s --- 600 --- Per-instance wall-clock limit (s)

Reference
---------
C. Expósito-Izquierdo, B. Melián-Batista, J.M. Moreno-Vega,
"An exact approach for the Blocks Relocation Problem",
Expert Systems with Applications 42 (2015) 6408–6422.
"""

from __future__ import annotations

from typing import Dict, Optional

from algorithms.common.tree import TreeSolver
from algorithms.common.tree.components import (
    BlockAssignment,
    LB2,
    RestrictedCRPSpace,
)
from algorithms.common.tree.strategies import BestFirstBB

from .rules import ExpositoGreedy, ExpositoPolicy


class ExposioBB2015(TreeSolver):
    """Expósito-Izquierdo et al. best-first B&B."""

    name = "Expósito-Izquierdo (2015) B&B"
    category = "Exact"
    description = "Expósito-Izquierdo et al. best-first branch-and-bound."
    compatible_problems = ["CRP-R"]
    geometry = "single-bay"
    objectives = ["relocations"]
    fidelity = "faithful"

    def build_search(self) -> BestFirstBB:
        """Compose the paper's best-first search."""
        extra = self.config.extra or {}
        raw_alpha = extra.get("alpha", 0)
        alpha: Optional[int]
        if raw_alpha in (0, "0", "inf", None):
            alpha = None
        else:
            alpha = max(1, int(raw_alpha))

        space = RestrictedCRPSpace()
        branching = BlockAssignment(space)
        lower_bound = LB2(space)
        greedy_policy = ExpositoPolicy(
            space,
            lower_bound,
            alpha=1,
        )
        return BestFirstBB(
            state_space=space,
            branching=branching,
            lower_bound=lower_bound,
            upper_bound=ExpositoGreedy(
                space,
                branching,
                greedy_policy,
            ),
            child_policy=ExpositoPolicy(
                space,
                lower_bound,
                alpha=alpha,
            ),
            complete_search=alpha is None,
        )

    @classmethod
    def config_schema(cls) -> Dict:
        schema = {}
        schema.update({
            "alpha": {
                "type": "int",
                "default": 0,
                "min": 0,
                "max": 100,
                "label": "alpha (0 = unlimited / optimal)",
            },
            "time_limit_s": {
                "type": "float",
                "default": 600.0,
                "min": 1.0,
                "max": 86400.0,
                "label": "Time limit (s)",
            },
        })
        return schema
