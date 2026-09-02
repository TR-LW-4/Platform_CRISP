"""
CRP-Online
<online> <lookahead> <CRP-Online>
Online Block Relocation Problem with progressive revelation

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP".
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, Set

import numpy as np

from problems.CRP_R import CRP_R


class CRP_Online(CRP_R):
    """Restricted CRP with progressive revelation of future priorities."""

    name = "CRP-Online"
    description = (
        "Restricted container relocation with progressively revealed retrieval "
        "order information."
    )
    tags = ["crp", "online", "fixed-order", "yard-only"]
    metric_names = ["relocations", "time", "steps"]

    def _lookahead_h(self) -> int:
        extra = self.config.extra or {}
        raw = int(extra.get("lookahead_h", 0) or 0)
        return max(0, raw)

    def _revealed_priorities(self) -> Set[int]:
        if self._done:
            return set()
        start = int(self._current_target_priority)
        end = min(
            int(self.config.num_containers),
            start + self._lookahead_h(),
        )
        return set(range(start, end + 1))

    def _get_obs(self) -> np.ndarray:
        arr = self.yard.get_state_array().copy()
        revealed = self._revealed_priorities()
        if revealed:
            mask = np.isin(arr[..., 1], list(revealed), assume_unique=False)
            arr[..., 1] = np.where(mask, arr[..., 1], 0)
        else:
            arr[..., 1] = 0
        target_pri = np.array([self._current_target_priority], dtype=np.int32)
        return np.concatenate([arr.flatten(), target_pri])

    def _get_info(self) -> Dict[str, Any]:
        info = super()._get_info()
        info["lookahead_h"] = float(self._lookahead_h())
        info["revealed_targets"] = float(len(self._revealed_priorities()))
        return info

    @classmethod
    def config_schema(cls) -> Dict[str, Dict]:
        schema = super().config_schema()
        schema.update(
            {
                "lookahead_h": {
                    "type": "int",
                    "default": 0,
                    "min": 0,
                    "max": 1000,
                    "label": "Look-ahead horizon H",
                    "help": (
                        "Number of future targets revealed in addition to the "
                        "current target. H=0 matches Zehendner et al. (2017)."
                    ),
                },
            }
        )
        return schema
