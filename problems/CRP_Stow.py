"""
P4 – CRP-Stow (Container Stowage Planning)

Based on the existing spp-main / cspp-main environments.

Rules
-----
- Containers are distributed in a yard (bays × rows × tiers).
- Each container belongs to a group (destination port).
- A vessel has slots organised by bay/row/tier; each slot is pre-assigned
  to a group.
- At each step the sequencer selects the next empty vessel slot.
- The agent chooses which yard container to load (must match the slot's
  group).  Any containers stacked above the chosen container must first
  be shifted down (shifters = relocations).
- Objective: minimise total shifters.

Gym interface
-------------
Observation : flat yard state + current vessel slot info
Action      : yard slot index (which container to pick from the yard)
Reward      : −shifters
Terminal    : all vessel slots of available groups filled
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym

from core.base_problem import BaseProblem, ProblemConfig
from core.container import Container, make_containers
from core.yard import Yard, Stack


class CRP_Stow(BaseProblem):

    name         = "CRP-Stow"
    description  = ("Container Stowage Planning (yard → vessel). "
                    "Load containers from yard onto vessel, minimising shifters.")
    tags         = ["relocation", "stowage", "vessel", "group-matching"]
    metric_names = ["shifters", "time", "vessel_utilisation"]

    def __init__(
        self,
        config: Optional[ProblemConfig] = None,
        render_mode: Optional[str]      = None,
    ):
        super().__init__(config, render_mode)
        self._total_shifters:    int            = 0
        self._vessel_filled:     int            = 0
        self._current_slot:      Optional[int]  = None   # flat vessel slot index
        self._available_groups:  np.ndarray     = np.array([], dtype=int)
        self._done:              bool           = False

        # Vessel state: shape (V, 5) – bay/row/tier/occupied/group
        self._vessel_state: Optional[np.ndarray] = None
        self._vessel_slots: int = 0

    # ---------------------------------------------------------------- #
    # Spaces                                                             #
    # ---------------------------------------------------------------- #

    def _setup_spaces(self) -> None:
        cfg = self.config
        n_yard_slots  = cfg.num_bays  * cfg.num_rows  * cfg.max_tiers
        n_vessel_slots = cfg.vessel_bays * cfg.vessel_rows * cfg.vessel_tiers

        # obs = flat yard (5 attrs each) + current vessel slot (5 attrs)
        obs_size = n_yard_slots * 5 + 5
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(obs_size,), dtype=np.int32
        )

        self._yard_action_n = n_yard_slots
        self.action_space   = gym.spaces.Discrete(n_yard_slots)

    # ---------------------------------------------------------------- #
    # Reset                                                              #
    # ---------------------------------------------------------------- #

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.config.seed = seed

        self.yard.clear()
        self._total_shifters = 0
        self._vessel_filled  = 0
        self._done           = False

        self._build_episode()

        self._current_slot = self._next_vessel_slot()
        if self._current_slot is None:
            self._done = True

        obs  = self._get_obs()
        info = self._get_info()
        return obs, info

    def _build_episode(self) -> None:
        cfg = self.config
        rng = np.random.RandomState(cfg.seed)

        # ── Yard containers ───────────────────────────────────────── #
        containers = make_containers(
            n=cfg.num_containers,
            num_groups=cfg.num_groups,
            seed=cfg.seed,
            enable_weight=cfg.enable_weight,
            enable_size=cfg.enable_size,
            enable_type=cfg.enable_type,
        )
        self.containers = containers

        order      = rng.permutation(len(containers)).tolist()
        stack_list = list(self.yard.stacks.keys())
        slot_i     = 0
        for idx in order:
            key = stack_list[slot_i % len(stack_list)]
            stk = self.yard.stacks[key]
            if not stk.is_full:
                self.yard.place(*key, containers[idx])
            slot_i += 1

        # ── Vessel state ──────────────────────────────────────────── #
        vB, vR, vT = cfg.vessel_bays, cfg.vessel_rows, cfg.vessel_tiers
        self._vessel_slots = vB * vR * vT
        # cols: [bay, row, tier, occupied, group]
        self._vessel_state = np.zeros((self._vessel_slots, 5), dtype=np.int32)
        idx = 0
        for b in range(1, vB + 1):
            for r in range(1, vR + 1):
                for t in range(1, vT + 1):
                    self._vessel_state[idx, 0] = b
                    self._vessel_state[idx, 1] = r
                    self._vessel_state[idx, 2] = t
                    idx += 1

        # Assign groups to vessel slots (evenly distributed)
        slots_per_group = self._vessel_slots // cfg.num_groups
        for g in range(cfg.num_groups):
            start = g * slots_per_group
            end   = (g + 1) * slots_per_group if g < cfg.num_groups - 1 else self._vessel_slots
            self._vessel_state[start:end, 4] = g

        # Available groups = groups present in yard
        yard_groups = set(c.group for c in self.containers)
        self._available_groups = np.array(sorted(yard_groups), dtype=int)

    # ---------------------------------------------------------------- #
    # Vessel sequencer                                                   #
    # ---------------------------------------------------------------- #

    def _next_vessel_slot(self) -> Optional[int]:
        """Return index of next empty vessel slot whose group is in available_groups."""
        if self._vessel_state is None:
            return None
        empty  = self._vessel_state[:, 3] == 0
        in_grp = np.isin(self._vessel_state[:, 4], self._available_groups)
        valid  = np.where(empty & in_grp)[0]
        if len(valid) == 0:
            return None
        # Sort by (bay, tier, row) – load bottom first
        bays  = self._vessel_state[valid, 0]
        tiers = self._vessel_state[valid, 2]
        rows  = self._vessel_state[valid, 1]
        order = np.lexsort((rows, tiers, bays))
        return int(valid[order[0]])

    # ---------------------------------------------------------------- #
    # Step                                                               #
    # ---------------------------------------------------------------- #

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self._done or self._current_slot is None:
            return self._get_obs(), 0.0, True, False, self._get_info()

        action     = int(action)
        target_grp = int(self._vessel_state[self._current_slot, 4])

        # Decode action → (bay, row, tier)
        bay, row, tier = self._slot_to_coords(action)
        key = (bay, row)
        stk = self.yard.stacks.get(key)

        reward = 0.0

        if (
            stk is None
            or tier < 1
            or tier > stk.height
            or stk.containers[tier - 1].group != target_grp
        ):
            reward = -0.5   # invalid
        else:
            container = stk.containers[tier - 1]
            blockers  = stk.blockers_above(container)
            n_shifters = len(blockers)

            # Shift blockers down (they cascade down one tier)
            if blockers:
                # Remove target first, then drop blockers
                stk.containers.remove(container)
                # Blockers are now lower; they stay in same stack, positions shift
                # (gravity: each blocker drops by 1 tier – no action needed,
                #  they're already at the right index after target removal)
            else:
                stk.pop()

            # Place container in vessel slot
            self._vessel_state[self._current_slot, 3] = 1
            self._vessel_state[self._current_slot, 4] = container.group
            self._vessel_filled += 1

            # Update available groups
            yard_groups = set(c.group for c in self.yard.all_containers())
            self._available_groups = np.array(sorted(yard_groups), dtype=int)

            self._total_shifters += n_shifters
            reward = -float(n_shifters)

            # Next vessel slot
            self._current_slot = self._next_vessel_slot()
            if self._current_slot is None or len(self._available_groups) == 0:
                self._done = True

        obs  = self._get_obs()
        info = self._get_info()
        return obs, reward, self._done, False, info

    # ---------------------------------------------------------------- #
    # Evaluate (for EA)                                                  #
    # ---------------------------------------------------------------- #

    def evaluate(self, solution: List[int]) -> Dict[str, float]:
        self.reset()
        for action in solution:
            if self._done:
                break
            self.step(action)
        return self.get_metrics()

    def get_metrics(self) -> Dict[str, float]:
        total_slots = self._vessel_slots if self._vessel_slots > 0 else 1
        return {
            "shifters":           float(self._total_shifters),
            "time":               float(self._total_shifters),
            "vessel_utilisation": self._vessel_filled / total_slots,
            "vessel_filled":      float(self._vessel_filled),
        }

    # ---------------------------------------------------------------- #
    # Helpers                                                            #
    # ---------------------------------------------------------------- #

    def _slot_to_coords(self, action: int) -> Tuple[int, int, int]:
        cfg   = self.config
        rows  = cfg.num_rows
        tiers = cfg.max_tiers
        bay   = action // (rows * tiers) + 1
        rem   = action  % (rows * tiers)
        row   = rem // tiers + 1
        tier  = rem  % tiers + 1
        return bay, row, tier

    def _get_obs(self) -> np.ndarray:
        yard_flat = self.yard.get_flat_obs()
        if self._current_slot is not None and self._vessel_state is not None:
            slot_info = self._vessel_state[self._current_slot].astype(np.int32)
        else:
            slot_info = np.zeros(5, dtype=np.int32)
        return np.concatenate([yard_flat, slot_info])

    def _get_info(self) -> Dict:
        info = self.get_metrics()
        info["action_mask"]    = self._build_action_mask()
        info["current_slot"]   = self._current_slot
        info["total_shifters"] = self._total_shifters
        return info

    def _build_action_mask(self) -> np.ndarray:
        n    = self._yard_action_n
        mask = np.zeros(n, dtype=np.bool_)
        if self._current_slot is None or self._vessel_state is None:
            return mask
        target_grp = int(self._vessel_state[self._current_slot, 4])
        for i in range(n):
            bay, row, tier = self._slot_to_coords(i)
            stk = self.yard.stacks.get((bay, row))
            if stk and tier <= stk.height:
                if stk.containers[tier - 1].group == target_grp:
                    mask[i] = True
        return mask

    @classmethod
    def config_schema(cls) -> Dict:
        schema = super().config_schema()
        schema.update({
            "vessel_bays":  {"type": "int", "default": 2, "min": 1, "max": 10},
            "vessel_rows":  {"type": "int", "default": 2, "min": 1, "max": 6},
            "vessel_tiers": {"type": "int", "default": 3, "min": 2, "max": 10},
        })
        return schema
