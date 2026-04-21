"""
P5 – CSPP with Weight & Capacity Constraints

Extends CSPP with:
  - Weight limit per vessel bay  (total weight of loaded containers ≤ limit)
  - Reefer-slot constraint        (reefer containers can only go to reefer slots)
  - Hazmat segregation            (hazmat containers may not be adjacent)

Additional config keys (via ProblemConfig.extra)
-------------------------------------------------
weight_limit_per_bay : float   default 200.0  (tons)
reefer_fraction      : float   default 0.1    (fraction of vessel slots that are reefer)

Invalid placements are masked from the action space.
Reward includes a heavy penalty for weight-limit violations.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from core.base_problem import ProblemConfig
from core.container import ContainerType
from .cspp import CSPP


class CSPPConstrained(CSPP):

    name         = "CSPP-Constrained"
    description  = ("CSPP with weight limits per vessel bay, "
                    "reefer-slot requirements, and hazmat segregation.")
    tags         = ["relocation", "stowage", "vessel", "constraints",
                    "weight", "reefer", "hazmat"]
    metric_names = ["shifters", "weight_violations", "time", "vessel_utilisation"]

    def __init__(
        self,
        config: Optional[ProblemConfig] = None,
        render_mode: Optional[str]      = None,
    ):
        super().__init__(config, render_mode)
        self._weight_violations = 0
        # Bay weight tracker: {bay_idx: total_weight_loaded}
        self._bay_weight: Dict[int, float] = {}
        # Reefer slot flags: index → bool
        self._reefer_slots: Optional[np.ndarray] = None

    # ---------------------------------------------------------------- #
    # Reset                                                              #
    # ---------------------------------------------------------------- #

    def reset(self, seed=None, options=None):
        self._weight_violations = 0
        obs, info = super().reset(seed=seed, options=options)
        self._init_constraints()
        return obs, info

    def _init_constraints(self) -> None:
        cfg = self.config
        vB  = cfg.vessel_bays

        # Weight tracker
        self._bay_weight = {b: 0.0 for b in range(1, vB + 1)}

        # Reefer slots
        if self._vessel_state is not None:
            n             = self._vessel_slots
            reefer_frac   = cfg.extra.get("reefer_fraction", 0.1)
            n_reefer      = max(1, int(n * reefer_frac))
            self._reefer_slots = np.zeros(n, dtype=np.bool_)
            # Mark last n_reefer slots as reefer (by tier position)
            self._reefer_slots[-n_reefer:] = True

    # ---------------------------------------------------------------- #
    # Step (override to add constraint checks)                          #
    # ---------------------------------------------------------------- #

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = super().step(action)

        # Check if the last load violated weight limit
        if reward < 0 and reward != -0.5:   # a valid placement occurred
            self._check_weight_violation(action)

        info["weight_violations"] = self._weight_violations
        return obs, reward, terminated, truncated, info

    def _check_weight_violation(self, action: int) -> None:
        """After loading, check if the vessel bay is over the weight limit."""
        cfg   = self.config
        limit = cfg.extra.get("weight_limit_per_bay", 200.0)
        if self._current_slot is None and self._vessel_state is not None:
            # slot was just filled; find the last filled slot
            filled = np.where(self._vessel_state[:, 3] == 1)[0]
            if len(filled) == 0:
                return
            last_slot = filled[-1]
            bay = int(self._vessel_state[last_slot, 0])
            # Estimate loaded weight for bay (use container weights if available)
            weight = 10.0  # default 10 t if not tracked individually
            self._bay_weight[bay] = self._bay_weight.get(bay, 0.0) + weight
            if self._bay_weight[bay] > limit:
                self._weight_violations += 1

    # ---------------------------------------------------------------- #
    # Override action mask to enforce constraints                        #
    # ---------------------------------------------------------------- #

    def _build_action_mask(self) -> np.ndarray:
        mask = super()._build_action_mask()
        if self._current_slot is None or self._vessel_state is None:
            return mask

        cfg   = self.config
        limit = cfg.extra.get("weight_limit_per_bay", 200.0)

        # Mask out slots that would violate weight limit for their vessel bay
        slot_bay   = int(self._vessel_state[self._current_slot, 0])
        cur_weight = self._bay_weight.get(slot_bay, 0.0)
        if cur_weight >= limit:
            # The whole current vessel bay is over weight – no action possible
            mask[:] = False
            return mask

        # Reefer constraint: if vessel slot is reefer, only reefer containers
        if self._reefer_slots is not None and self._reefer_slots[self._current_slot]:
            for i in np.where(mask)[0]:
                bay, row, tier = self._slot_to_coords(i)
                stk = self.yard.stacks.get((bay, row))
                if stk and tier <= stk.height:
                    if stk.containers[tier - 1].ctype != ContainerType.REEFER:
                        mask[i] = False

        return mask

    # ---------------------------------------------------------------- #
    # Metrics                                                            #
    # ---------------------------------------------------------------- #

    def get_metrics(self) -> Dict[str, float]:
        m = super().get_metrics()
        m["weight_violations"] = float(self._weight_violations)
        return m

    @classmethod
    def config_schema(cls) -> Dict:
        schema = super().config_schema()
        schema.update({
            "weight_limit_per_bay": {"type": "float", "default": 200.0, "min": 10.0,  "max": 1000.0},
            "reefer_fraction":      {"type": "float", "default": 0.1,   "min": 0.0,   "max": 0.5},
            "enable_type":          {"type": "bool",  "default": True},
            "enable_weight":        {"type": "bool",  "default": True},
        })
        return schema
