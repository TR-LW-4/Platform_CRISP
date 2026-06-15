"""
CRP-D – Duplicate-priority unrestricted Block Relocation Problem.

Semantics
---------
- Containers are assigned to **groups** (duplicate priorities).  All containers
  in group 1 must leave before group 2, etc. — but within a group the retrieval
  order is free (any accessible container of the current group may be taken).
- Relocation is **unrestricted**: the crane may move the **top container of
  any stack** to any other non-full stack (same rule as CRP-U).
- Objective: minimise total relocations.

Benchmark files
---------------
When ``extra["layout_file_path"]`` points to a ZhuDup ``.txt`` file the
episode loads containers from that file.  Otherwise a random duplicate-priority
layout is generated using ``config.num_groups`` distinct group IDs spread over
``config.num_containers`` containers.

This class is completely standalone.  It imports nothing from CRP_U or
CRP_Stow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from core.base_problem import BaseProblem, ProblemConfig
from core.container import Container, ContainerSize, ContainerType
from core.layout_trace import trace_layout
from core.benchmark_keys import layout_path_from_extra


class CRP_D(BaseProblem):
    """
    Duplicate-priority unrestricted BRP.

    Action space : Discrete(S * S) where S = num_bays * num_rows.
                   Action k = src_idx * S + dst_idx (any top → any non-full).
    Observation  : flat yard tensor (S * max_tiers * 5 attrs) + current target
                   group ID (1 scalar) → length S*T*5 + 1.
    Reward       : −1 per relocation, 0 otherwise; terminal when all groups
                   retrieved.
    """

    name        = "CRP-D"
    description = (
        "Duplicate-priority unrestricted BRP. "
        "Groups of containers must be retrieved in group-ID order; within a "
        "group any accessible container counts.  Any stack-top may be relocated "
        "(unrestricted).  Loads ZhuDup benchmark files when "
        "``layout_file_path`` is set; otherwise uses random duplicate layout."
    )
    tags         = ["crp", "duplicate", "unrestricted", "fixed-group-order", "yard-only"]
    metric_names = ["relocations", "steps", "time"]

    # ---------------------------------------------------------------- #
    # Init                                                               #
    # ---------------------------------------------------------------- #

    def __init__(
        self,
        config: Optional[ProblemConfig] = None,
        render_mode: Optional[str]      = None,
    ):
        super().__init__(config, render_mode)
        self._total_relocations: int = 0
        self._total_steps:       int = 0
        self._done:              bool = False
        self._current_group_idx: int = 0   # index into self._sorted_groups
        self._sorted_groups:     List[int] = []

    # ---------------------------------------------------------------- #
    # Spaces                                                             #
    # ---------------------------------------------------------------- #

    def _setup_spaces(self) -> None:
        cfg     = self.config
        n       = cfg.num_bays * cfg.num_rows
        obs_size = n * cfg.max_tiers * 5 + 1
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(obs_size,), dtype=np.int32
        )
        self.action_space = gym.spaces.Discrete(n * n)

    # ---------------------------------------------------------------- #
    # Reset                                                              #
    # ---------------------------------------------------------------- #

    def reset(
        self,
        seed:    Optional[int]  = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.config.seed = seed

        self.yard.clear()
        self._total_relocations = 0
        self._total_steps       = 0
        self._done              = False
        self._current_group_idx = 0

        self._build_episode()

        # Cache sorted unique group IDs once for this episode.
        self._sorted_groups = sorted(set(c.priority for c in self.containers))

        opts = options or {}
        if opts.get("skip_auto_retrieve"):
            return self._get_obs(), self._get_info()

        self._advance_auto_retrievals()
        return self._get_obs(), self._get_info()

    # ---------------------------------------------------------------- #
    # Episode builder                                                    #
    # ---------------------------------------------------------------- #

    def _build_episode(self) -> None:
        path_str = layout_path_from_extra(self.config.extra)
        if path_str:
            self._load_zhu_dup_file(Path(path_str))
        else:
            self._build_random_episode()

    def _load_zhu_dup_file(self, path: Path) -> None:
        from core.zhu_dup_benchmark import apply_dup_file_to_yard

        trace_layout(f"CRP_D._build_episode: ZhuDup file {path.resolve()}")
        self.containers = apply_dup_file_to_yard(self.yard, self.config, path)

    def _build_random_episode(self) -> None:
        cfg = self.config
        rng = np.random.RandomState(cfg.seed)

        n, k = cfg.num_containers, max(1, cfg.num_groups)
        # Distribute containers across groups as evenly as possible.
        base, extra_cnt = divmod(n, k)
        group_ids: List[int] = []
        for g in range(1, k + 1):
            cnt = base + (1 if g <= extra_cnt else 0)
            group_ids.extend([g] * cnt)
        rng.shuffle(group_ids)

        containers: List[Container] = []
        for i, gid in enumerate(group_ids):
            containers.append(Container(
                id=i + 1,
                group=gid,
                priority=gid,
                weight=10.0,
                size=ContainerSize.TEU,
                ctype=ContainerType.STANDARD,
            ))
        self.containers = containers

        order      = rng.permutation(len(containers)).tolist()
        stack_keys = list(self.yard.stacks.keys())
        slot       = 0
        for idx in order:
            if slot >= len(stack_keys) * cfg.max_tiers:
                break
            key = stack_keys[slot % len(stack_keys)]
            stk = self.yard.stacks[key]
            if not stk.is_full:
                self.yard.place(*key, containers[idx])
            slot += 1

    # ---------------------------------------------------------------- #
    # Auto-retrievals                                                    #
    # ---------------------------------------------------------------- #

    def _current_target_group(self) -> Optional[int]:
        if self._current_group_idx >= len(self._sorted_groups):
            return None
        return self._sorted_groups[self._current_group_idx]

    def _group_remaining_in_yard(self, group_id: int) -> int:
        """Count containers of *group_id* still sitting in any stack."""
        total = 0
        for stk in self.yard.stacks.values():
            for c in stk.containers:
                if c.priority == group_id:
                    total += 1
        return total

    def _advance_auto_retrievals(self) -> None:
        """Retrieve every accessible target-group container until blocked."""
        changed = True
        while changed:
            changed = False
            tg = self._current_target_group()
            if tg is None:
                self._done = True
                return
            for stk in self.yard.stacks.values():
                if stk.is_empty:
                    continue
                if stk.top.priority == tg:
                    stk.pop()
                    self.yard.total_retrievals += 1
                    if self._group_remaining_in_yard(tg) == 0:
                        self._current_group_idx += 1
                        if self._current_group_idx >= len(self._sorted_groups):
                            self._done = True
                            return
                    changed = True
                    break   # restart outer while to re-evaluate target group

    # ---------------------------------------------------------------- #
    # Step                                                               #
    # ---------------------------------------------------------------- #

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self._done:
            return self._get_obs(), 0.0, True, False, self._get_info()

        self._total_steps += 1
        n       = self.config.num_bays * self.config.num_rows
        a       = int(action) % (n * n)
        src_idx = a // n
        dst_idx = a  % n
        src     = self._idx_to_stack(src_idx)
        dst     = self._idx_to_stack(dst_idx)

        src_stk = self.yard.stacks.get(src)
        dst_stk = self.yard.stacks.get(dst)
        tg      = self._current_target_group()
        reward  = 0.0

        if (
            src == dst
            or src_stk is None or src_stk.is_empty
            or dst_stk is None or dst_stk.is_full
        ):
            reward = -0.5
        elif tg is not None and src_stk.top.priority == tg:
            # Agent chose to relocate the target itself — penalise.
            reward = -0.5
        else:
            self.yard.relocate(src, dst)
            self._total_relocations += 1
            reward = -1.0

        self._advance_auto_retrievals()
        return self._get_obs(), reward, self._done, False, self._get_info()

    # ---------------------------------------------------------------- #
    # Evaluate (for EA / batch planners)                                #
    # ---------------------------------------------------------------- #

    def evaluate(self, solution: List[int]) -> Dict[str, float]:
        saved_seed = self.config.seed
        self.reset()
        for action in solution:
            if self._done:
                break
            self.step(action)
        self.config.seed = saved_seed
        return self.get_metrics()

    # ---------------------------------------------------------------- #
    # Metrics                                                            #
    # ---------------------------------------------------------------- #

    def get_metrics(self) -> Dict[str, float]:
        total_groups = max(len(self._sorted_groups), 1)
        return {
            "relocations": float(self._total_relocations),
            "steps":       float(self._total_steps),
            "time":        float(self._total_relocations),
            "progress":    self._current_group_idx / total_groups,
        }

    # ---------------------------------------------------------------- #
    # Observation / info / mask                                         #
    # ---------------------------------------------------------------- #

    def _get_obs(self) -> np.ndarray:
        flat   = self.yard.get_flat_obs()
        tg     = self._current_target_group()
        target = np.array([tg if tg is not None else 0], dtype=np.int32)
        return np.concatenate([flat, target])

    def _get_info(self) -> Dict:
        info = self.get_metrics()
        info["action_mask"]     = self._build_action_mask()
        info["target_group"]    = self._current_target_group()
        info["current_group_idx"] = self._current_group_idx
        return info

    def _build_action_mask(self) -> np.ndarray:
        n    = self.config.num_bays * self.config.num_rows
        mask = np.zeros(n * n, dtype=np.bool_)
        tg   = self._current_target_group()
        for si in range(n):
            src = self._idx_to_stack(si)
            ss  = self.yard.stacks.get(src)
            if ss is None or ss.is_empty:
                continue
            if tg is not None and ss.top.priority == tg:
                continue   # do not relocate accessible target
            for di in range(n):
                if si == di:
                    continue
                dst = self._idx_to_stack(di)
                ds  = self.yard.stacks.get(dst)
                if ds is not None and not ds.is_full:
                    mask[si * n + di] = True
        return mask

    # ---------------------------------------------------------------- #
    # Helpers                                                            #
    # ---------------------------------------------------------------- #

    def _idx_to_stack(self, idx: int) -> Tuple[int, int]:
        nr  = self.config.num_rows
        idx = int(idx) % (self.config.num_bays * nr)
        return idx // nr + 1, idx % nr + 1

    # ---------------------------------------------------------------- #
    # Config schema                                                      #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict:
        schema = super().config_schema()
        return schema
