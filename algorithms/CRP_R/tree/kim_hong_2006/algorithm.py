"""
KimHong2006BB
<2006> <exact> <restricted> <single-bay> <CRP-R>
Depth-first branch-and-bound with block-assignment branches
time_limit_s --- 300 --- Per-instance wall-clock limit (s)

Reference
---------
K.H. Kim, G.P. Hong,
"A heuristic rule for relocating blocks",
Computers & Operations Research 33 (2006) 940–954.
"""

from __future__ import annotations

from typing import Dict

from algorithms.common.tree import TreeSolver
from algorithms.common.tree.components import (
    BlockAssignment,
    LB1,
    RestrictedCRPSpace,
)
from algorithms.common.tree.strategies import DepthFirstBB


class KimHong2006BB(TreeSolver):
    """Kim and Hong's exact depth-first branch-and-bound."""

    name = "Kim & Hong (2006) B&B"
    category = "Exact"
    description = "Kim & Hong (COR 2006) exact branch-and-bound."
    compatible_problems = ["CRP-R"]
    geometry = "single-bay"
    objectives = ["relocations"]
    fidelity = "faithful"

    def build_search(self) -> DepthFirstBB:
        """Compose the paper's block-assignment B&B."""
        space = RestrictedCRPSpace()
        return DepthFirstBB(
            state_space=space,
            branching=BlockAssignment(space),
            lower_bound=LB1(space),
        )

    @classmethod
    def config_schema(cls) -> Dict:
        schema = {}
        schema.update({
            "time_limit_s": {
                "type": "float",
                "default": 300.0,
                "min": 1.0,
                "max": 86400.0,
                "label": "Time limit (s)",
            },
        })
        return schema
