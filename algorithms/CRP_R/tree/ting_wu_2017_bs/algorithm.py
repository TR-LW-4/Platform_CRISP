"""
TingWuBS
<2017> <heuristic> <restricted> <single-bay> <CRP-R>
Beam search with virtual relocation heuristic
beam_width --- 5 --- Nodes retained at each level
evaluator --- VRH --- Evaluation heuristic: VRH or SDH

Reference
---------
C.-J. Ting, K.-C. Wu,
"Optimizing container relocation operations at container yards with beam search",
Transportation Research Part E 103 (2017) 17–31.
"""

from __future__ import annotations

from typing import Dict

from algorithms.common.tree import TreeSolver
from algorithms.common.tree.components import (
    RestrictedCRPSpace,
    SingleMove,
)
from algorithms.common.tree.strategies import BeamSearch

from .scoring import TingEvaluator


class TingWuBS(TreeSolver):
    """Ting and Wu's beam search."""

    name = "Ting & Wu (2017) BS"
    category = "Heuristic"
    description = "Ting & Wu (TRE 2017) beam search with VRH."
    compatible_problems = ["CRP-R"]
    geometry = "single-bay"
    objectives = ["relocations"]
    fidelity = "faithful"

    def build_search(self) -> BeamSearch:
        """Compose beam search and the selected paper evaluator."""
        extra = self.config.extra or {}
        width = max(1, int(extra.get("beam_width", 5)))
        evaluator = str(extra.get("evaluator", "VRH")).upper()

        space = RestrictedCRPSpace()
        return BeamSearch(
            state_space=space,
            branching=SingleMove(space),
            evaluator=TingEvaluator(space, evaluator),
            width=width,
            diverse_parents=True,
        )

    @classmethod
    def config_schema(cls) -> Dict:
        schema = {}
        schema.update({
            "beam_width": {
                "type": "int",
                "default": 5,
                "min": 1,
                "max": 50,
                "label": "Beam width",
            },
            "evaluator": {
                "type": "str",
                "default": "VRH",
                "choices": ["VRH", "SDH"],
                "label": "Evaluation heuristic",
            },
            "time_limit_s": {
                "type": "float",
                "default": 300.0,
                "min": 1.0,
                "max": 86400.0,
                "label": "Time limit (s)",
            },
        })
        return schema
