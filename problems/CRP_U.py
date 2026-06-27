"""
CRP-U – unrestricted *relocations* (Jovanović, Tuba & Voß, EJOR 2019).

Semantics (aligned with literature rBRP vs uBRP)
------------------------------------------------
- **Retrieval order is fixed**: only the current smallest due-date / priority
  container may leave the yard (strict 1 … N), same as CRP-R.
- **Unrestricted** refers to **which container may be relocated** in the next
  move: the crane may move the **top container of any stack** to any other
  non-full stack — not only blockers sitting on the **target** stack
  (restricted BRP).

This is **not** “free choice of retrieval order” (where the agent freely
chooses which container to retrieve next).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from problems.CRP_R import CRP_R


class CRP_U(CRP_R):
    """
    uBRP on the platform: fixed retrieval sequence + unrestricted relocation
    candidate set (any stack top → any non-full stack).
    """

    name = "CRP-U"
    description = (
        "Unrestricted BRP (Jovanović et al., EJOR 2019): fixed retrieval order "
        "1…N; each step may relocate the top container of **any** stack. "
        "Contrasts with CRP-R / restricted BRP, where only the top blocker on "
        "the target's stack may be moved."
    )
    tags         = ["crp", "unrestricted", "unrestricted-brp", "fixed-order", "yard-only"]
    metric_names = ["relocations", "steps", "time"]

    hide_from_problem_list = False

    # ---------------------------------------------------------------- #
    # Spaces: (src_stack, dst_stack) pairs                              #
    # ---------------------------------------------------------------- #

    def _num_stacks(self) -> int:
        return self.config.num_bays * self.config.num_rows

    def _setup_spaces(self) -> None:
        n   = self.config.num_bays * self.config.num_rows
        obs_size = self.config.num_bays * self.config.num_rows * self.config.max_tiers * 5 + 1
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(obs_size,), dtype=np.int32
        )
        self.action_space = gym.spaces.Discrete(n * n)

    def _pair_decode(self, action: int) -> Tuple[int, int]:
        n = self._num_stacks()
        a = int(action) % (n * n)
        return a // n, a % n

    # ---------------------------------------------------------------- #
    # Step: relocate top of *src* stack to *dst*                         #
    # ---------------------------------------------------------------- #

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self._done:
            return self._get_obs(), 0.0, True, False, self._get_info()

        self._total_steps += 1

        src_i, dst_i = self._pair_decode(action)
        src = self._action_to_stack(src_i)
        dst = self._action_to_stack(dst_i)
        reward = 0.0

        src_stack = self.yard.stacks.get(src)
        dst_stack = self.yard.stacks.get(dst)

        if (
            src == dst
            or src_stack is None
            or src_stack.is_empty
            or dst_stack is None
            or dst_stack.is_full
        ):
            reward = -0.5
        else:
            target = self._get_target_container()
            top    = src_stack.top
            # If current target is on top here, auto-retrievals should already
            # have fired; do not relocate the target container away.
            if target is not None and top is not None and top.id == target.id:
                reward = -0.5
            else:
                self.yard.relocate(src, dst)
                self._hook_after_relocate(src, dst, top.id)
                self._total_relocations += 1
                reward = -1.0

        self._advance_auto_retrievals()

        obs  = self._get_obs()
        info = self._get_info()
        return obs, reward, self._done, False, info

    def _build_action_mask(self) -> np.ndarray:
        n    = self._num_stacks()
        mask = np.zeros(n * n, dtype=np.bool_)
        target = self._get_target_container()
        for src_i in range(n):
            for dst_i in range(n):
                if src_i == dst_i:
                    continue
                s_key = self._action_to_stack(src_i)
                d_key = self._action_to_stack(dst_i)
                s_stk = self.yard.stacks.get(s_key)
                d_stk = self.yard.stacks.get(d_key)
                if (
                    s_stk is None
                    or s_stk.is_empty
                    or d_stk is None
                    or d_stk.is_full
                ):
                    continue
                if target is not None and s_stk.top is not None:
                    if s_stk.top.id == target.id:
                        continue
                mask[src_i * n + dst_i] = True
        return mask

    # ---------------------------------------------------------------- #
    # Evaluate sequential policy (flat pair indices)                   #
    # ---------------------------------------------------------------- #

    def evaluate(self, solution: List[int]) -> Dict[str, float]:
        """``solution[k]`` = flat pair index ``src * n_stacks + dst``."""
        saved_seed = self.config.seed
        self.reset()
        total_reloc = 0
        for action in solution:
            if self._done:
                break
            _, reward, _, _, _ = self.step(action)
            if reward <= -1.0:
                total_reloc += 1
        self.config.seed = saved_seed
        return {
            "relocations": float(total_reloc),
            "steps":       float(self._total_steps),
            "time":        float(total_reloc),
        }
