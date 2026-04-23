"""
P1 – Block Relocation Problem (Fixed Retrieval Order)

Rules
-----
- Containers in a yard have pre-assigned priorities 1 … N.
- Must retrieve in strict order: priority 1 first, then 2, … then N.
- When the target is buried, every container above it must be relocated
  to another stack before the target can be retrieved.
- Objective: minimise total relocations.

Gym interface
-------------
Observation : flat yard state + current target priority (normalised)
Action      : destination stack index (bay×row, flattened) for the
              topmost blocker.  When target is already on top the
              environment retrieves automatically with no action needed.
Reward      : 0 at each step; −1 terminal penalty per relocation.
             (dense variant: −1 every relocation during episode)

Plan interface (for batch planning algorithms, e.g. Lee & Lee 2010)
--------------------------------------------------------------------
evaluate_plan(plan: RelocationPlan) → metrics dict
  Simulates a complete RelocationPlan on a fresh episode and returns
  relocations, crane_time (if kinematics configured), and other metrics.
  Kinematics parameters are read from ProblemConfig.extra:
    gantry_s_per_bay  (default 3.5 s)
    trolley_s_per_row (default 1.2 s)
    gantry_accel_s    (default 40 s)
    spreader_s        (default 30 s)
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym

from core.base_problem import BaseProblem, ProblemConfig
from core.container import Container, make_containers
from core.yard import Yard
from core.plan import RelocationPlan, simulate_plan
from core.objectives import KinematicsModel, compute_crane_time, lower_bound_relocations


class BRPFixed(BaseProblem):

    name         = "BRP-Fixed"
    description  = ("Block Relocation Problem with fixed (known) retrieval order. "
                    "Minimise total relocations.")
    tags         = ["relocation", "fixed-order", "yard-only"]
    metric_names = ["relocations", "time", "steps"]

    def __init__(
        self,
        config: Optional[ProblemConfig] = None,
        render_mode: Optional[str]      = None,
    ):
        super().__init__(config, render_mode)
        self._current_target_priority: int = 1
        self._total_relocations: int       = 0
        self._total_steps: int             = 0
        self._done: bool                   = False

    # ---------------------------------------------------------------- #
    # Episode hooks (override in subclasses, e.g. CRP-Time)              #
    # ---------------------------------------------------------------- #

    def _hooks_clear_episode(self) -> None:
        """Called in reset() after yard.clear(), before _build_episode()."""

    def _hook_after_relocate(
        self,
        src: Tuple[int, int],
        dst: Tuple[int, int],
        container_id: int,
    ) -> None:
        """Called after a successful blocker relocation in step()."""

    def _hook_after_retrieve(self, bay: int, row: int, container_id: int) -> None:
        """Called after each automatic retrieval in _advance_auto_retrievals()."""

    # ---------------------------------------------------------------- #
    # Spaces                                                             #
    # ---------------------------------------------------------------- #

    def _setup_spaces(self) -> None:
        cfg = self.config
        n_stacks = cfg.num_bays * cfg.num_rows

        # Observation: flat yard (bays×rows×tiers × 5 attrs) + target priority
        obs_size = cfg.num_bays * cfg.num_rows * cfg.max_tiers * 5 + 1
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(obs_size,), dtype=np.int32
        )

        # Action: choose which destination stack to send the blocker
        self.action_space = gym.spaces.Discrete(n_stacks)

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
        self._hooks_clear_episode()

        self._build_episode()

        self._current_target_priority = 1
        self._advance_auto_retrievals()

        obs  = self._get_obs()
        info = self._get_info()
        return obs, info

    def _build_episode(self) -> None:
        """Fill yard with containers having priorities 1…N in random positions."""
        cfg = self.config
        rng = np.random.RandomState(cfg.seed)

        containers = make_containers(
            n=cfg.num_containers,
            num_groups=1,           # BRP uses priority, not group
            seed=cfg.seed,
            enable_weight=cfg.enable_weight,
            enable_size=cfg.enable_size,
            enable_type=cfg.enable_type,
            priority_assignment="sequential",
        )
        self.containers = containers

        # Shuffle containers and place them into stacks row by row
        order = rng.permutation(len(containers)).tolist()
        stack_list = list(self.yard.stacks.keys())

        slot = 0
        for idx in order:
            if slot >= len(stack_list) * cfg.max_tiers:
                break
            stack_key = stack_list[slot % len(stack_list)]
            stk = self.yard.stacks[stack_key]
            if not stk.is_full:
                self.yard.place(*stack_key, containers[idx])
            slot += 1

    # ---------------------------------------------------------------- #
    # Auto-retrieve: pull targets that are already on top               #
    # ---------------------------------------------------------------- #

    def _advance_auto_retrievals(self) -> None:
        """Retrieve all consecutive targets that are already accessible."""
        while True:
            target = self._get_target_container()
            if target is None:
                self._done = True
                return
            stack = self.yard._find_stack(target)
            if stack is None:
                self._done = True
                return
            if stack.top == target:
                stack.pop()
                self._hook_after_retrieve(stack.bay, stack.row, target.id)
                self.yard.total_retrievals += 1
                self._current_target_priority += 1
            else:
                return   # blocked – wait for agent action

    # ---------------------------------------------------------------- #
    # Step                                                               #
    # ---------------------------------------------------------------- #

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self._done:
            obs = self._get_obs()
            return obs, 0.0, True, False, self._get_info()

        self._total_steps += 1

        # Decode action → destination stack key
        dst = self._action_to_stack(action)

        # Identify the target and its topmost blocker
        target = self._get_target_container()
        if target is None:
            self._done = True
            return self._get_obs(), 0.0, True, False, self._get_info()

        src_stack = self.yard._find_stack(target)
        if src_stack is None or src_stack.top == target:
            # Should have been auto-retrieved; skip gracefully
            self._advance_auto_retrievals()
            obs  = self._get_obs()
            info = self._get_info()
            return obs, 0.0, self._done, False, info

        # Relocate the topmost blocker to chosen destination
        src = (src_stack.bay, src_stack.row)
        dst_stack = self.yard.stacks.get(dst)
        reward = 0.0

        if dst_stack is None or dst_stack.is_full or dst == src:
            # Invalid action → penalise slightly
            reward = -0.5
        else:
            blocker = src_stack.top
            self.yard.relocate(src, dst)
            self._hook_after_relocate(src, dst, blocker.id)
            self._total_relocations += 1
            reward = -1.0

        # After relocation, auto-retrieve what is now accessible
        self._advance_auto_retrievals()

        obs  = self._get_obs()
        info = self._get_info()
        return obs, reward, self._done, False, info

    # ---------------------------------------------------------------- #
    # Evaluate (for EA)                                                  #
    # ---------------------------------------------------------------- #

    def evaluate(self, solution: List[int]) -> Dict[str, float]:
        """
        Simulate the solution on a fresh episode.
        solution[i] = destination stack index for the i-th relocation step.
        """
        saved_seed = self.config.seed
        self.reset()
        total_reloc = 0
        for action in solution:
            if self._done:
                break
            _, reward, _, _, _ = self.step(action)
            if reward <= -1.0:
                total_reloc += 1
        # Restore
        self.config.seed = saved_seed
        return {"relocations": total_reloc, "steps": self._total_steps, "time": float(total_reloc)}

    # ---------------------------------------------------------------- #
    # Metrics                                                            #
    # ---------------------------------------------------------------- #

    def get_metrics(self) -> Dict[str, float]:
        return {
            "relocations": float(self._total_relocations),
            "steps":       float(self._total_steps),
            "time":        float(self._total_relocations),   # 1 reloc = 1 time unit
            "progress":    self._current_target_priority / max(self.config.num_containers, 1),
        }

    # ---------------------------------------------------------------- #
    # Helpers                                                            #
    # ---------------------------------------------------------------- #

    def _get_target_container(self) -> Optional[Container]:
        for c in self.containers:
            if c.priority == self._current_target_priority:
                return c
        return None

    def _action_to_stack(self, action: int) -> Tuple[int, int]:
        """Map flat action index → (bay, row)."""
        action = int(action) % (self.config.num_bays * self.config.num_rows)
        bay = action // self.config.num_rows + 1
        row = action  % self.config.num_rows + 1
        return (bay, row)

    def _get_obs(self) -> np.ndarray:
        flat = self.yard.get_flat_obs()
        target_pri = np.array(
            [self._current_target_priority], dtype=np.int32
        )
        return np.concatenate([flat, target_pri])

    def _get_info(self) -> Dict:
        info = self.get_metrics()
        info["action_mask"] = self._build_action_mask()
        return info

    def _build_action_mask(self) -> np.ndarray:
        n = self.config.num_bays * self.config.num_rows
        mask = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            key = self._action_to_stack(i)
            stk = self.yard.stacks.get(key)
            if stk and not stk.is_full:
                mask[i] = True
        return mask

    # ---------------------------------------------------------------- #
    # Plan-based evaluation (for batch planning algorithms)             #
    # ---------------------------------------------------------------- #

    def evaluate_plan(self, plan: RelocationPlan) -> Dict[str, float]:
        """
        Evaluate a complete RelocationPlan on a fresh episode.

        Unlike evaluate() which takes a step-wise action-index list,
        this method accepts a RelocationPlan (container_id + positions)
        as produced by planning algorithms such as Lee & Lee (2010).

        Kinematics (crane working time) are computed when RMGC parameters
        are present in ProblemConfig.extra; otherwise crane_time = relocations.

        Returns
        -------
        dict with keys: relocations, crane_time, total_moves, lower_bound,
                        lb_ratio, steps, feasible
        """
        self.reset()
        kin    = KinematicsModel.from_config_extra(self.config.extra)
        result = simulate_plan(self.yard, plan, self.config.max_tiers)
        lb     = lower_bound_relocations(self.yard)

        if result.feasible:
            crane_time = compute_crane_time(plan, kin)
        else:
            crane_time = float("inf")

        return {
            "relocations": float(result.num_relocations),
            "crane_time":  crane_time,
            "total_moves": float(result.num_moves),
            "lower_bound": float(lb),
            "lb_ratio":    float(result.num_relocations / max(lb, 1)),
            "steps":       float(result.num_moves),
            "feasible":    float(result.feasible),
            "time":        crane_time,
        }

    # ---------------------------------------------------------------- #
    # Config schema                                                      #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict:
        schema = super().config_schema()
        # Kinematics parameters for RMGC time model (Lee & Lee 2010 defaults).
        # Keys not in ProblemConfig are automatically stored in config.extra by GUI.
        schema.update({
            "gantry_s_per_bay": {
                "type": "float", "default": 3.5, "min": 0.5, "max": 20.0,
                "label": "Gantry speed (s/bay)",
                "help": "Gantry travel time per bay (Lee & Lee default: 3.5 s).",
            },
            "trolley_s_per_row": {
                "type": "float", "default": 1.2, "min": 0.1, "max": 10.0,
                "label": "Trolley speed (s/row)",
                "help": "Trolley travel time per container width (default: 1.2 s).",
            },
            "gantry_accel_s": {
                "type": "float", "default": 40.0, "min": 0.0, "max": 120.0,
                "label": "Gantry accel overhead (s)",
                "help": "Combined acceleration + deceleration when gantry moves (default: 40 s).",
            },
            "spreader_s": {
                "type": "float", "default": 30.0, "min": 5.0, "max": 120.0,
                "label": "Spreader time (s)",
                "help": "Combined pickup + place-down time (default: 30 s).",
            },
        })
        return schema
