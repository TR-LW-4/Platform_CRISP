"""
Stochastic-CRP primitives shared by all CRP-Stoch algorithms.

Modules
-------
policy_tree        – Chance / Decision nodes for the Stochastic CRP (SCRP)
batch_layout       – Batch generation helpers (Bacci 2022 style)
galle_2017_source  – Ported reference implementation (bay representation,
                     lower bounds, EG/EM/ERI/L/Rand policies, PBFS/PBFSA
                     tree search) from Galle et al.'s SCRP repository.
                     Import directly as ``core.stoch.galle_2017_source``.
"""

from __future__ import annotations

from .policy_tree import (
    ChanceNode,
    DecisionNode,
    PolicyTree,
    expected_relocations,
)
from .batch_layout import (
    make_batches,
    priority_to_batch_map,
    enumerate_realizations,
)

__all__ = [
    "ChanceNode",
    "DecisionNode",
    "PolicyTree",
    "expected_relocations",
    "make_batches",
    "priority_to_batch_map",
    "enumerate_realizations",
]
