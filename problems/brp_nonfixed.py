"""
P2 – Block Relocation Problem (Non-Fixed / Free Retrieval Order)

Rules
-----
- All containers must eventually be retrieved, but the agent is FREE
  to choose which container to retrieve next.
- Objective: minimise total relocations across ALL retrievals.
- The agent gains by choosing containers that are already accessible
  (on top of their stacks) first.

Gym interface
-------------
Observation : flat yard state
Action      : flat slot index (bay, row, tier) of the container to retrieve.
              The environment moves all blockers above it automatically
              and counts them as relocations.
Reward      : −(number of relocations triggered by this action)
Terminal    : yard is empty
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym

from core.base_problem import BaseProblem, ProblemConfig
from core.container import make_containers
from core.yard import Yard


class BRPNonFixed(BaseProblem):

    name         = "BRP-NonFixed"
    description  = ("Block Relocation Problem with free retrieval order. "
                    "Choose which container to retrieve next to minimise total relocations.")
    tags         = ["relocation", "non-fixed-order", "yard-only"]
    metric_names = ["relocations", "steps", "time"]
    # Base for CRP-U; omitted from GUI / ``list_problems()`` (see ``core.registry``).
    hide_from_problem_list = True

    def __init__(
        self,
        config: Optional[ProblemConfig] = None,
        render_mode: Optional[str]      = None,
    ):
        super().__init__(config, render_mode)
        self._total_relocations = 0
        self._total_steps       = 0
        self._done              = False

    # ---------------------------------------------------------------- #
    # Spaces                                                             #
    # ---------------------------------------------------------------- #

    def _setup_spaces(self) -> None:
        cfg = self.config
        n_slots = cfg.num_bays * cfg.num_rows * cfg.max_tiers

        obs_size = cfg.num_bays * cfg.num_rows * cfg.max_tiers * 5
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(obs_size,), dtype=np.int32
        )
        self.action_space = gym.spaces.Discrete(n_slots)

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
        self._total_relocations = 0
        self._total_steps       = 0
        self._done              = False

        self._build_episode()

        obs  = self._get_obs()
        info = self._get_info()
        return obs, info

    def _build_episode(self) -> None:
        cfg = self.config
        rng = np.random.RandomState(cfg.seed)

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
        slot       = 0
        for idx in order:
            if slot >= len(stack_list) * cfg.max_tiers:
                break
            key = stack_list[slot % len(stack_list)]
            stk = self.yard.stacks[key]
            if not stk.is_full:
                self.yard.place(*key, containers[idx])
            slot += 1

    # ---------------------------------------------------------------- #
    # Step                                                               #
    # ---------------------------------------------------------------- #

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self._done:
            return self._get_obs(), 0.0, True, False, self._get_info()

        self._total_steps += 1
        bay, row, tier = self._action_to_coords(action)
        key  = (bay, row)
        stk  = self.yard.stacks.get(key)

        reward = 0.0

        if stk is None or tier > stk.height or stk.is_empty:
            reward = -0.5   # invalid action penalty
        else:
            target = stk.containers[tier - 1]
            n_relocs, _ = self.yard.retrieve(target)
            self._total_relocations += n_relocs
            reward = -float(n_relocs)

        self._done = self.yard.is_empty()
        obs  = self._get_obs()
        info = self._get_info()
        return obs, reward, self._done, False, info

    # ---------------------------------------------------------------- #
    # Evaluate (for EA)                                                  #
    # ---------------------------------------------------------------- #

    def evaluate(self, solution: List[int]) -> Dict[str, float]:
        self.reset()
        total_reloc = 0
        for action in solution:
            if self._done:
                break
            _, reward, _, _, _ = self.step(action)
            total_reloc += max(0, int(-reward))
        return {
            "relocations": float(total_reloc),
            "steps":       float(self._total_steps),
            "time":        float(total_reloc),
        }

    def get_metrics(self) -> Dict[str, float]:
        return {
            "relocations": float(self._total_relocations),
            "steps":       float(self._total_steps),
            "time":        float(self._total_relocations),
        }

    # ---------------------------------------------------------------- #
    # Helpers                                                            #
    # ---------------------------------------------------------------- #

    def _action_to_coords(self, action: int) -> Tuple[int, int, int]:
        cfg  = self.config
        rows = cfg.num_rows
        tiers = cfg.max_tiers
        bay  = action // (rows * tiers) + 1
        rem  = action  % (rows * tiers)
        row  = rem // tiers + 1
        tier = rem  % tiers + 1
        return bay, row, tier

    def _get_obs(self) -> np.ndarray:
        return self.yard.get_flat_obs()

    def _get_info(self) -> Dict:
        info = self.get_metrics()
        info["action_mask"] = self._build_action_mask()
        return info

    def _build_action_mask(self) -> np.ndarray:
        cfg   = self.config
        n     = cfg.num_bays * cfg.num_rows * cfg.max_tiers
        mask  = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            bay, row, tier = self._action_to_coords(i)
            stk = self.yard.stacks.get((bay, row))
            if stk and tier <= stk.height:
                mask[i] = True
        return mask
