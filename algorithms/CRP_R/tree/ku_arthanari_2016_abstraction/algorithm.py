"""
KuArthanari2016Abstraction
<2016> <exact> <restricted> <single-bay> <CRP-R>
Abstraction method with pattern-database bidirectional search
time_limit_s --- 300 --- Per-instance wall-clock limit (s)
pdb_depth --- 5 --- PDB remaining-container depth

Reference
---------
D. Ku, T.S. Arthanari,
"On the abstraction method for the container relocation problem",
Computers & Operations Research 68 (2016) 110–122.
"""

from __future__ import annotations

from typing import Dict

from algorithms.common.tree import TreeSolver
from algorithms.common.tree.components import RestrictedCRPSpace

from .search import KuAbstractionSearch


class KuArthanari2016Abstraction(TreeSolver):
    """Ku and Arthanari's abstraction and PDB search."""

    name = "Ku & Arthanari (2016) Abstraction+PDB"
    category = "Exact"
    description = "Ku & Arthanari abstraction + pattern-database search."
    compatible_problems = ["CRP-R"]
    geometry = "single-bay"
    objectives = ["relocations"]
    fidelity = "faithful"

    def build_search(self) -> KuAbstractionSearch:
        """Configure the paper-specific abstraction engine."""
        extra = self.config.extra or {}
        return KuAbstractionSearch(
            state_space=RestrictedCRPSpace(),
            pdb_depth=int(extra.get("pdb_depth", 5)),
            cache_depth=int(extra.get("cache_depth", 20)),
            max_cache_size=int(
                extra.get("max_cache_size", 1_000_000)
            ),
            pdb_build_time_limit_s=float(
                extra.get("pdb_build_time_limit_s", 60.0)
            ),
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
            "pdb_depth": {
                "type": "int",
                "default": 5,
                "min": 0,
                "max": 14,
                "label": "PDB depth",
            },
            "cache_depth": {
                "type": "int",
                "default": 20,
                "min": 0,
                "max": 200,
                "label": "Forward cache depth",
            },
            "max_cache_size": {
                "type": "int",
                "default": 1_000_000,
                "min": 1_000,
                "max": 20_000_000,
                "label": "Max forward-cache entries",
            },
            "pdb_build_time_limit_s": {
                "type": "float",
                "default": 60.0,
                "min": 1.0,
                "max": 3600.0,
                "label": "PDB build time limit (s)",
            },
        })
        return schema
