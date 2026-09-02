"""
CRP-Prem
<premarshalling> <distinct> <CRP-Prem>
Container Pre-Marshalling Problem (no retrievals)

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP".
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym

from core.base_problem import BaseProblem, ProblemConfig
from core.container import make_containers


class CRP_Prem(BaseProblem):

    name         = "CRP-Prem"
    description  = "Pre-marshalling containers into a well-ordered yard layout."
    tags         = ["relocation", "pre-marshalling", "yard-only", "sorting"]
    metric_names = ["moves", "bad_overlaps", "time"]

    def __init__(
        self,
        config: Optional[ProblemConfig] = None,
        render_mode: Optional[str]      = None,
    ):
        super().__init__(config, render_mode)
        self._total_moves = 0
        self._done        = False

    # ---------------------------------------------------------------- #
    # Spaces                                                             #
    # ---------------------------------------------------------------- #

    def _setup_spaces(self) -> None:
        cfg     = self.config
        n_stacks = cfg.num_bays * cfg.num_rows

        obs_size = cfg.num_bays * cfg.num_rows * cfg.max_tiers * 5 + 1
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(obs_size,), dtype=np.int32
        )
        # Action: (src_stack_idx, dst_stack_idx) packed as a single int
        self.action_space = gym.spaces.Discrete(n_stacks * (n_stacks - 1))

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
        self._total_moves = 0
        self._done        = self.yard.is_sorted()

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
        )
        self.containers = containers

        # Assign consecutive priorities within each group
        by_group: Dict[int, List] = {}
        for c in containers:
            by_group.setdefault(c.group, []).append(c)
        counter = 1
        for g in sorted(by_group):
            for c in by_group[g]:
                c.priority = counter
                counter += 1

        order      = rng.permutation(len(containers)).tolist()
        stack_list = list(self.yard.stacks.keys())
        slot       = 0
        for idx in order:
            key = stack_list[slot % len(stack_list)]
            stk = self.yard.stacks[key]
            if not stk.is_full:
                self.yard.place(*key, containers[idx])
            slot += 1

        self._done = self.yard.is_sorted()

    # ---------------------------------------------------------------- #
    # Step                                                               #
    # ---------------------------------------------------------------- #

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self._done:
            return self._get_obs(), 0.0, True, False, self._get_info()

        src_idx, dst_idx = self._decode_action(action)
        src = self._idx_to_stack(src_idx)
        dst = self._idx_to_stack(dst_idx)

        src_stk = self.yard.stacks.get(src)
        dst_stk = self.yard.stacks.get(dst)

        reward = 0.0
        if src_stk is None or dst_stk is None or src_stk.is_empty or dst_stk.is_full or src == dst:
            reward = -0.5
        else:
            self.yard.relocate(src, dst)
            self._total_moves += 1
            reward = -1.0

        self._done = self.yard.is_sorted()
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
        return {
            "moves":        float(self._total_moves),
            "bad_overlaps": float(self.yard.total_bad_overlaps()),
            "time":         float(self._total_moves),
        }

    def get_metrics(self) -> Dict[str, float]:
        return {
            "moves":        float(self._total_moves),
            "bad_overlaps": float(self.yard.total_bad_overlaps()),
            "time":         float(self._total_moves),
        }

    # ---------------------------------------------------------------- #
    # Helpers                                                            #
    # ---------------------------------------------------------------- #

    def _decode_action(self, action: int) -> Tuple[int, int]:
        n = self.config.num_bays * self.config.num_rows
        action = int(action) % (n * (n - 1))
        src = action // (n - 1)
        dst = action  % (n - 1)
        if dst >= src:
            dst += 1
        return src, dst

    def _idx_to_stack(self, idx: int) -> Tuple[int, int]:
        row_n = self.config.num_rows
        bay   = idx // row_n + 1
        row   = idx  % row_n + 1
        return (bay, row)

    def _get_obs(self) -> np.ndarray:
        flat = self.yard.get_flat_obs()
        return np.concatenate([flat, [self.yard.total_bad_overlaps()]])

    def _get_info(self) -> Dict:
        info = self.get_metrics()
        info["action_mask"] = self._build_action_mask()
        return info

    def _build_action_mask(self) -> np.ndarray:
        n    = self.config.num_bays * self.config.num_rows
        size = n * (n - 1)
        mask = np.zeros(size, dtype=np.bool_)
        for a in range(size):
            src_i, dst_i = self._decode_action(a)
            src = self._idx_to_stack(src_i)
            dst = self._idx_to_stack(dst_i)
            ss  = self.yard.stacks.get(src)
            ds  = self.yard.stacks.get(dst)
            if ss and ds and not ss.is_empty and not ds.is_full:
                mask[a] = True
        return mask
