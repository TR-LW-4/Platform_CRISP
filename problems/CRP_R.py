"""
CRP-R
<restricted> <distinct> <CRP-R>
Classical Block Relocation Problem with fixed retrieval order

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP".
--------------------------------------------------------------------------
"""

from __future__ import annotations

from numbers import Integral
from typing import Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym

from core.base_problem import BaseProblem, ProblemConfig
from core.container import Container, make_containers
from core.yard import Move, Yard
from core.plan import Movement, RelocationPlan
from core.objectives import KinematicsModel, compute_crane_time, lower_bound_relocations
from core.layout_trace import trace_layout
from core.benchmark_keys import layout_path_from_extra


class CRP_R(BaseProblem):

    name         = "CRP-R"
    description = (
        "Restricted container relocation with fixed retrieval order; "
        "only blockers above the current target may be moved."
    )
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
        self._last_validation_errors: List[str] = []

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

    def _setup_spaces(self) -> None:
        cfg = self.config
        n_stacks = cfg.num_bays * cfg.num_rows
        obs_size = cfg.num_bays * cfg.num_rows * cfg.max_tiers * 5 + 1
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(obs_size,), dtype=np.int32
        )
        self.action_space = gym.spaces.Discrete(n_stacks)

    # ---------------------------------------------------------------- #
    # Reset                                                              #
    # ---------------------------------------------------------------- #

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Load a layout and start retrieval from priority 1.

        ``options["skip_auto_retrieve"]``: keep every container after load so
        opening retrievals can appear explicitly in a plan.
        """
        super().reset(seed=seed)
        if seed is not None:
            self.config.seed = seed

        self.yard.clear()
        self._total_relocations = 0
        self._total_steps       = 0
        self._done              = False
        self._hooks_clear_episode()

        self._build_episode()
        opts = options if options is not None else {}
        if opts.get("skip_auto_retrieve"):
            self._current_target_priority = 1
            obs  = self._get_obs()
            info = self._get_info()
            return obs, info
        return self._finish_reset_after_layout_loaded()

    def _finish_reset_after_layout_loaded(self) -> Tuple[np.ndarray, Dict]:
        """Set the retrieval cursor and take any targets already on stack tops."""
        self._current_target_priority = 1

        self._advance_auto_retrievals()

        obs  = self._get_obs()
        info = self._get_info()
        return obs, info

    def _build_episode(self) -> None:
        """Fill yard from ``extra['layout_file_path']`` (Caserta/Zhu-style file).

        CRP-R / CRP-U / CRP-Time require a layout file. Subclasses such as
        CRP-MO and CRP-Stoch still fall back to a random yard when no file is set.
        """
        cfg = self.config
        path_str = layout_path_from_extra(cfg.extra)
        if path_str:
            from pathlib import Path

            from core.caserta_benchmark import apply_caserta_file_to_yard

            trace_layout(f"CRP_R._build_episode: read **1** layout file from disk (each reset ⇒ one file): {path_str}")
            p = Path(path_str)
            self.containers = apply_caserta_file_to_yard(self.yard, cfg, p)
            return

        if self.name in ("CRP-R", "CRP-U", "CRP-Time"):
            raise ValueError(
                f"{self.name} requires a Caserta/Zhu layout file "
                "(set extra['layout_file_path']). Random layouts are disabled."
            )

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
                self.yard.move_history.append(
                    Move("retrieve", target.id, (stack.bay, stack.row), None)
                )
                self._hook_after_retrieve(stack.bay, stack.row, target.id)
                self.yard.total_retrievals += 1
                self._current_target_priority += 1
            else:
                return   # blocked — wait for the next relocation

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

    def evaluate(self, solution: List[int]) -> Dict[str, float]:
        """Replay destination indices; invalid moves are skipped, not rejected."""
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
        return {"relocations": total_reloc, "steps": self._total_steps, "time": float(total_reloc)}

    def validate_actions(self, solution: List[int]) -> Dict[str, float]:
        """
        Strictly replay a complete restricted-BRP destination sequence.

        Unlike :meth:`evaluate`, this method is intended as the common
        publication-facing validator.  Invalid, extra, or incomplete action
        sequences are rejected instead of being silently scored as partial
        solutions.
        """
        saved_seed = self.config.seed
        self._last_validation_errors = []

        self.reset(options={"skip_auto_retrieve": True})
        lb = lower_bound_relocations(self.yard)
        self._finish_reset_after_layout_loaded()

        n_stacks = self.config.num_bays * self.config.num_rows
        for step_index, raw_action in enumerate(solution):
            if self._done:
                self._last_validation_errors.append(
                    f"action {step_index}: extra action after all containers were retrieved"
                )
                break
            if isinstance(raw_action, bool) or not isinstance(raw_action, Integral):
                self._last_validation_errors.append(
                    f"action {step_index}: destination index must be an integer"
                )
                break

            action = int(raw_action)
            if action < 0 or action >= n_stacks:
                self._last_validation_errors.append(
                    f"action {step_index}: destination index {action} outside [0, {n_stacks})"
                )
                break

            target = self._get_target_container()
            src_stack = self.yard._find_stack(target) if target is not None else None
            if src_stack is None or src_stack.top == target:
                self._last_validation_errors.append(
                    f"action {step_index}: no blocker is available to relocate"
                )
                break

            src = (src_stack.bay, src_stack.row)
            dst = self._action_to_stack(action)
            dst_stack = self.yard.stacks.get(dst)
            if dst == src:
                self._last_validation_errors.append(
                    f"action {step_index}: destination equals source {src}"
                )
                break
            if dst_stack is None:
                self._last_validation_errors.append(
                    f"action {step_index}: destination stack {dst} does not exist"
                )
                break
            if dst_stack.is_full:
                self._last_validation_errors.append(
                    f"action {step_index}: destination stack {dst} is full"
                )
                break

            before = self._total_relocations
            self.step(action)
            if self._total_relocations != before + 1:
                self._last_validation_errors.append(
                    f"action {step_index}: relocation was not executed"
                )
                break

        completed = self._done
        if not completed and not self._last_validation_errors:
            self._last_validation_errors.append(
                "action sequence ended before all containers were retrieved"
            )

        feasible = completed and not self._last_validation_errors
        plan = RelocationPlan([
            Movement(container_id=m.container_id, from_pos=m.src, to_pos=m.dst)
            for m in self.yard.move_history
            if m.kind in ("relocate", "retrieve")
        ])
        crane_time = (
            compute_crane_time(
                plan, KinematicsModel.from_config_extra(self.config.extra)
            )
            if feasible
            else float("inf")
        )
        executed_relocations = float(self._total_relocations)
        relocations = executed_relocations if feasible else float("inf")
        self.config.seed = saved_seed
        return {
            "relocations": relocations,
            "executed_relocations": executed_relocations,
            "crane_time": crane_time,
            "total_moves": float(plan.num_moves()),
            "lower_bound": float(lb),
            "lb_ratio": float(relocations / max(lb, 1)),
            "steps": float(self._total_steps),
            "feasible": float(feasible),
            "completed": float(completed),
            "validation_conflicts": float(len(self._last_validation_errors)),
            "validated": 1.0,
            "time": crane_time,
        }

    def get_last_validation_errors(self) -> List[str]:
        """Return human-readable errors from the most recent strict validation."""
        return list(self._last_validation_errors)

    # ---------------------------------------------------------------- #
    # Metrics                                                            #
    # ---------------------------------------------------------------- #

    def get_metrics(self) -> Dict[str, float]:
        return {
            "relocations": float(self._total_relocations),
            "steps":       float(self._total_steps),
            "time":        float(self._total_relocations),
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

    def evaluate_plan(self, plan: RelocationPlan) -> Dict[str, float]:
        """Alias for :meth:`validate_plan`."""
        return self.validate_plan(plan)

    def validate_plan(self, plan: RelocationPlan) -> Dict[str, float]:
        """
        Strictly validate a complete explicit plan under CRP-R rules.

        Every retrieval must follow priority order and every relocation must
        move the top blocker from the *current target's* stack.  Consequently,
        cleaning moves from unrelated stacks are rejected for CRP-R.
        """
        saved_seed = self.config.seed
        self._last_validation_errors = []
        self.reset(options={"skip_auto_retrieve": True})
        lb = lower_bound_relocations(self.yard)
        relocations = 0
        retrievals = 0
        executed_moves = 0
        expected_priority = 1

        for step_index, move in enumerate(plan.movements):
            if expected_priority > self.config.num_containers:
                self._last_validation_errors.append(
                    f"move {step_index}: extra move after all containers were retrieved"
                )
                break

            src_stack = self.yard.stacks.get(move.from_pos)
            if src_stack is None or src_stack.is_empty:
                self._last_validation_errors.append(
                    f"move {step_index}: source stack {move.from_pos} is empty or missing"
                )
                break
            if src_stack.top.id != move.container_id:
                self._last_validation_errors.append(
                    f"move {step_index}: container {move.container_id} is not on top of "
                    f"{move.from_pos}"
                )
                break

            container = src_stack.top
            if move.is_retrieval:
                if container.priority != expected_priority:
                    self._last_validation_errors.append(
                        f"move {step_index}: expected priority {expected_priority}, "
                        f"got {container.priority}"
                    )
                    break
                src_stack.pop()
                self.yard.move_history.append(
                    Move("retrieve", container.id, move.from_pos, None)
                )
                self.yard.total_retrievals += 1
                self._hook_after_retrieve(
                    move.from_pos[0], move.from_pos[1], container.id
                )
                expected_priority += 1
                retrievals += 1
                executed_moves += 1
                continue

            target = next(
                (c for c in self.containers if c.priority == expected_priority),
                None,
            )
            target_stack = self.yard._find_stack(target) if target is not None else None
            target_pos = (
                (target_stack.bay, target_stack.row)
                if target_stack is not None
                else None
            )
            if target_pos is None or move.from_pos != target_pos:
                self._last_validation_errors.append(
                    f"move {step_index}: relocation is not from current target stack "
                    f"{target_pos}"
                )
                break
            if container == target:
                self._last_validation_errors.append(
                    f"move {step_index}: current target must be retrieved, not relocated"
                )
                break
            if move.to_pos == move.from_pos:
                self._last_validation_errors.append(
                    f"move {step_index}: destination equals source {move.from_pos}"
                )
                break
            dst_stack = self.yard.stacks.get(move.to_pos)
            if dst_stack is None:
                self._last_validation_errors.append(
                    f"move {step_index}: destination stack {move.to_pos} does not exist"
                )
                break
            if dst_stack.is_full:
                self._last_validation_errors.append(
                    f"move {step_index}: destination stack {move.to_pos} is full"
                )
                break

            self.yard.relocate(move.from_pos, move.to_pos)
            self._hook_after_relocate(
                move.from_pos, move.to_pos, container.id
            )
            self._total_relocations += 1
            relocations += 1
            executed_moves += 1

        completed = (
            expected_priority > self.config.num_containers
            and all(stack.is_empty for stack in self.yard.stacks.values())
        )
        if not completed and not self._last_validation_errors:
            self._last_validation_errors.append(
                "plan ended before all containers were retrieved"
            )
        feasible = completed and not self._last_validation_errors
        self._current_target_priority = expected_priority
        self._done = completed
        self._total_steps = relocations

        crane_time = (
            compute_crane_time(
                plan, KinematicsModel.from_config_extra(self.config.extra)
            )
            if feasible
            else float("inf")
        )
        self.config.seed = saved_seed
        official_relocations = float(relocations) if feasible else float("inf")
        return {
            "relocations": official_relocations,
            "executed_relocations": float(relocations),
            "crane_time": crane_time,
            "total_moves": float(executed_moves),
            "retrievals": float(retrievals),
            "lower_bound": float(lb),
            "lb_ratio": float(official_relocations / max(lb, 1)),
            "steps": float(executed_moves),
            "feasible": float(feasible),
            "completed": float(completed),
            "validation_conflicts": float(len(self._last_validation_errors)),
            "validated": 1.0,
            "time": crane_time,
        }

    # ---------------------------------------------------------------- #
    # Config schema                                                      #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict:
        schema = super().config_schema()
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
