"""
CRP-Online (2026 Paper)
<online> <lookahead> <2026-paper> <isolated>
Self-contained implementation of Đurasević et al. (2026) Online CRP.

------------------------------- Reference --------------------------------
Marko Đurasević, Mateja Đumić, Francisco Javier Gil Gala,
"Automated design of relocation rules with genetic programming for the online
 container relocation problem",
Expert Systems with Applications (2026). DOI: 10.1016/j.eswa.2026.132786.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from core.base_problem import BaseProblem, ProblemConfig
import crp_online.core.environment as isolated_env
import crp_online.core.instance as isolated_inst
import crp_online.core.observation as isolated_obs
from crp_online.config.defaults import INT_MAX_SENTINEL
from crp_online.io.caserta_loader import parse_caserta_file
from crp_online.io.ku_loader import parse_ku_as_problem_instance


def _resolve_layout_file(config: ProblemConfig) -> Optional[Path]:
    """Resolves benchmark layout file path from config.extra if provided."""
    extra = config.extra or {}
    raw = (
        extra.get("layout_file_path")
        or extra.get("caserta_dat_path")
        or extra.get("layout_file")
    )
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_file() else None


def _generate_synthetic_instance(
    num_stacks: int,
    num_containers: int,
    max_tiers: int,
    seed: int,
) -> isolated_inst.ProblemInstance:
    """Generates a reproducible, non-empty initial CRP layout with guaranteed relocation headroom."""
    rng = random.Random(seed)
    containers = list(range(1, num_containers + 1))
    rng.shuffle(containers)

    stacks_layout: List[List[int]] = [[] for _ in range(num_stacks)]
    for idx, c in enumerate(containers):
        s = idx % num_stacks
        stacks_layout[s].append(c)

    initial_height = max(len(stk) for stk in stacks_layout) if stacks_layout else 1
    # Mathematical capacity guarantee: remaining S-1 stacks must hold all N containers
    # (S - 1) * H >= N  =>  H >= ceil(N / (S - 1)) + 2 buffer tiers
    min_feasible_h = (num_containers + max(1, num_stacks - 1) - 1) // max(1, num_stacks - 1) + 2
    resolved_max_tiers = max(max_tiers, initial_height + 2, min_feasible_h)
    return isolated_inst.ProblemInstance(
        name=f"synthetic_S{num_stacks}_N{num_containers}_seed{seed}",
        num_stacks=num_stacks,
        num_containers=num_containers,
        max_stack_height=resolved_max_tiers,
        initial_height=initial_height,
        initial_layout=tuple(tuple(stk) for stk in stacks_layout),
    )


class CRP_Online_2026(BaseProblem):
    """
    100% self-contained Problem Adapter for Đurasević et al. (2026).
    Enforces pure-decision Gymnasium state with automatic post-transition retrieval.
    Zero dependency on legacy core.yard.Yard or problems.CRP_Stoch.
    Supports Rolling Lookahead Horizon, Container Groups (Normal & Relaxed),
    and both Restricted and Unrestricted Relocation Schemes.
    """

    name: str = "CRP-Online (2026 Paper)"
    description: str = (
        "Isolated, verified implementation of Durasevic et al. (2026) Online CRP. "
        "Supports rolling lookahead horizons, normal/relaxed container groups, "
        "and restricted/unrestricted relocation schemes with zero information leakage."
    )
    tags: List[str] = ["crp", "online", "2026-paper", "isolated", "yard-only"]
    metric_names: List[str] = ["relocations", "retrievals", "crane_time_s"]

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        config: Optional[ProblemConfig] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__(config=config, render_mode=render_mode)
        self._parse_paper_config()
        self._build_episode()

    @classmethod
    def config_schema(cls) -> Dict[str, Dict[str, Any]]:
        """
        Return the exact parameter schema for Đurasević et al. (2026) Online CRP.
        Excludes all non-relevant multi-bay and crane attributes.
        """
        return {
            "relocation_scheme": {
                "type": "str",
                "default": "restricted",
                "options": ["restricted", "unrestricted"],
                "label": "Relocation Scheme",
                "help": (
                    "Restricted RS (Algorithm 1): only moves blocking containers above target. "
                    "Unrestricted RS (Algorithm 2): performs preventive relocations from candidate destination."
                ),
            },
            "info_structure": {
                "type": "str",
                "default": "rolling_horizon",
                "options": ["rolling_horizon", "container_groups"],
                "label": "Information Structure",
                "help": "Approach to container knowledge: Rolling Lookahead Horizon or Container Groups.",
            },
            # --- Rolling Horizon Parameters ---
            "lookahead_horizon": {
                "type": "float",
                "default": 1.0,
                "options": [1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0],
                "label": "Lookahead Horizon (α)",
                "help": "Lookahead ratio α: 1.0 = offline problem (all containers known). 0.0 = only target known.",
                "visibleWhen": {"key": "info_structure", "values": ["rolling_horizon"]},
            },
            "unknown_treatment": {
                "type": "str",
                "default": "maximum",
                "options": ["maximum", "ignore"],
                "label": "Unknown Container Strategy",
                "help": "Maximum: proxy = max_known + 1; Ignore: sentinel value.",
                "visibleWhen": {"key": "info_structure", "values": ["rolling_horizon"]},
            },
            # --- Container Group Parameters ---
            "group_mode": {
                "type": "str",
                "default": "normal",
                "options": ["normal", "relaxed"],
                "label": "Group Information Mode",
                "help": (
                    "Normal: only group order/proxy is known; container numbers within group are unknown. "
                    "Relaxed: container numbers within the active target group are known."
                ),
                "visibleWhen": {"key": "info_structure", "values": ["container_groups"]},
            },
            "group_by": {
                "type": "str",
                "default": "group_count",
                "options": ["group_count", "group_size"],
                "label": "Group Definition",
                "help": "Partition containers by Group Count (GC) or Group Size (GS).",
                "visibleWhen": {"key": "info_structure", "values": ["container_groups"]},
            },
            "group_count": {
                "type": "int",
                "default": 2,
                "options": [2, 3, 4, 5],
                "label": "Group Count (GC)",
                "help": "Number of groups into which containers are partitioned.",
                "visibleWhen": [
                    {"key": "info_structure", "values": ["container_groups"]},
                    {"key": "group_by", "values": ["group_count"]},
                ],
            },
            "group_size": {
                "type": "int",
                "default": 2,
                "options": [2, 3, 4, 5],
                "label": "Group Size (GS)",
                "help": "Number of containers per group.",
                "visibleWhen": [
                    {"key": "info_structure", "values": ["container_groups"]},
                    {"key": "group_by", "values": ["group_size"]},
                ],
            },
            # --- Procedural / Synthetic Yard Parameters ---
            "num_rows": {
                "type": "int",
                "default": 5,
                "min": 3,
                "max": 10,
                "label": "Number of Stacks (S)",
                "help": "Number of stacks for synthetic yard generation.",
            },
            "max_tiers": {
                "type": "int",
                "default": 4,
                "min": 2,
                "max": 10,
                "label": "Max Stack Height (H)",
                "help": "Maximum tiers/height for synthetic yard generation.",
            },
            "num_containers": {
                "type": "int",
                "default": 12,
                "min": 1,
                "max": 200,
                "label": "Containers (N)",
                "help": "Total number of containers for synthetic yard generation.",
            },
            "seed": {
                "type": "int",
                "default": 42,
                "min": 0,
                "max": 9999,
                "label": "Generation Seed",
                "help": "Random seed for reproducible synthetic layout generation.",
            },
        }

    def _parse_paper_config(self) -> None:
        """Parses paper-specific parameters from problem config extra dict."""
        extra = self.config.extra or {}
        self.relocation_scheme: str = str(extra.get("relocation_scheme", "restricted")).lower()
        self.info_structure: str = str(extra.get("info_structure", "rolling_horizon")).lower()
        self.lookahead_horizon: float = float(extra.get("lookahead_horizon", 1.0))
        self.unknown_treatment: str = str(extra.get("unknown_treatment", "maximum")).lower()
        self.group_mode: str = str(extra.get("group_mode", "normal")).lower()
        self.group_by: str = str(extra.get("group_by", "group_count")).lower()
        self.group_count: int = int(extra.get("group_count", 2))
        self.group_size: int = int(extra.get("group_size", 2))

    def _setup_spaces(self) -> None:
        S = max(1, int(self.config.num_rows))
        min_feasible_h = (int(self.config.num_containers) + max(1, S - 1) - 1) // max(1, S - 1) + 2
        H = max(int(self.config.max_tiers), min_feasible_h)
        # Action chooses candidate destination stack [0, S - 1]
        self.action_space = spaces.Discrete(S)
        # Observation grid covers [0, S - 1] and [0, H - 1], with proxy sentinels up to 2^31 - 1
        self.observation_space = spaces.Box(
            low=-1.0,
            high=2147483647.0,
            shape=(H, S),
            dtype=np.float64,
        )

    def _build_episode(self) -> None:
        layout_path = _resolve_layout_file(self.config)
        if layout_path is not None:
            if layout_path.suffix == ".txt" or "ku_arthanari" in str(layout_path):
                self._instance = parse_ku_as_problem_instance(layout_path, seed=self.config.seed)
            else:
                self._instance = parse_caserta_file(layout_path)
        else:
            self._instance = _generate_synthetic_instance(
                num_stacks=self.config.num_rows,
                num_containers=self.config.num_containers,
                max_tiers=self.config.max_tiers,
                seed=self.config.seed,
            )
        # Adapt spaces to actual loaded instance
        S = self._instance.num_stacks
        H = self._instance.max_stack_height
        self.action_space = spaces.Discrete(S)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=2147483647.0,
            shape=(H, S),
            dtype=np.float64,
        )
        self._sim = isolated_env.Environment(self._instance, store_moves=True)

    def get_observation(self) -> isolated_obs.ObservationState:
        """Constructs active ObservationState enforcing the 2026 information barrier."""
        use_groups = (self.info_structure == "container_groups")
        known_within = (self.group_mode == "relaxed")
        g_size = self.group_size if self.group_by == "group_size" else None
        g_count = self.group_count if self.group_by == "group_count" else 2

        return isolated_obs.ObservationState(
            env=self._sim,
            look_ahead_level=self.lookahead_horizon,
            horizon_strategy=self.unknown_treatment,
            use_container_groups=use_groups,
            num_groups=g_count,
            group_size=g_size,
            known_within_group=known_within,
            mode="cpp_compat",
        )

    def _auto_retrieve(self) -> None:
        """Automatically retrieves targets that reach the top of their stack."""
        while not self._sim.is_empty():
            target = self._sim.get_current_target()
            target_stack = self._sim.get_stack_of_container(target)
            if self._sim.get_top_container(target_stack) == target:
                self._sim.retrieve(target_stack)
            else:
                break  # Target blocked: environment is in a decision state

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed, options=options)
        self._parse_paper_config()
        self._build_episode()
        self._auto_retrieve()
        return self._get_obs_grid(), {"action_mask": self._get_action_mask()}

    def _execute_preventive_relocations(self, src_stack: int, dest_stack: int) -> None:
        """Executes Algorithm 2 preventive relocations to clear smaller containers from destination stack."""
        while True:
            if self._sim.get_stack_height(dest_stack) == 0:
                break

            obs = self.get_observation()
            sc_top = obs.get_top_obs_value(src_stack)
            dc_top = obs.get_top_obs_value(dest_stack)

            if sc_top is not None and dc_top is not None and dc_top < sc_top:
                best_s1: Optional[int] = None
                lowest_index: float = float(INT_MAX_SENTINEL)

                for s1 in range(self._sim.num_stacks):
                    if s1 == src_stack or s1 == dest_stack or self._sim.is_stack_full(s1):
                        continue

                    s1_lowest = obs.get_stack_min_obs_value(s1)
                    if s1_lowest > dc_top:
                        if lowest_index > s1_lowest:
                            lowest_index = s1_lowest
                            best_s1 = s1
                        elif s1_lowest == float(INT_MAX_SENTINEL):
                            lowest_index = 0.0
                            best_s1 = s1

                if best_s1 is not None:
                    self._sim.relocate(dest_stack, best_s1)
                    continue

            break

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self._sim.is_empty():
            return self._get_obs_grid(), 0.0, True, False, self.get_metrics()

        target = self._sim.get_current_target()
        src_stack = self._sim.get_stack_of_container(target)
        dest_stack = int(action)

        # Illegal action check: moving onto source stack, full stack, or out of bounds
        if (
            dest_stack == src_stack
            or dest_stack < 0
            or dest_stack >= self._sim.num_stacks
            or self._sim.is_stack_full(dest_stack)
        ):
            # Penalty applied to primary metric
            self._sim.total_relocations += 10000
            return self._get_obs_grid(), -10000.0, True, False, self.get_metrics()

        # Unrestricted relocation scheme: execute preventive relocations from dest_stack
        if self.relocation_scheme == "unrestricted":
            self._execute_preventive_relocations(src_stack, dest_stack)

        # Execute physical relocation
        self._sim.relocate(src_stack, dest_stack)

        # Post-transition automatic retrieval loop
        self._auto_retrieve()

        terminated = self._sim.is_empty()
        reward = 1.0 if terminated else -1.0
        return self._get_obs_grid(), reward, terminated, False, self.get_metrics()

    def _get_action_mask(self) -> np.ndarray:
        """Returns boolean mask of valid candidate destination stacks."""
        mask = np.ones(self._sim.num_stacks, dtype=bool)
        if self._sim.is_empty():
            return np.zeros(self._sim.num_stacks, dtype=bool)

        target = self._sim.get_current_target()
        src = self._sim.get_stack_of_container(target)
        mask[src] = False
        for s in range(self._sim.num_stacks):
            if self._sim.is_stack_full(s):
                mask[s] = False
        return mask

    def _get_obs_grid(self) -> np.ndarray:
        """Emits observed values respecting the information-hiding barrier."""
        H, S = self.observation_space.shape
        grid = np.full((H, S), -1.0, dtype=np.float64)
        obs = self.get_observation()

        for s in range(min(S, self._sim.num_stacks)):
            for t in range(obs.get_stack_height(s)):
                if t < H:
                    grid[t, s] = obs.get_container(s, t).obs_value
        return grid

    def get_metrics(self) -> Dict[str, float]:
        return {
            "relocations": float(self._sim.total_relocations),
            "retrievals": float(self._sim.total_retrievals),
            "crane_time_s": float(self._sim.total_execution_time),
        }

    def evaluate(self, solution: List[int]) -> Dict[str, float]:
        self.reset()
        for act in solution:
            _, _, terminated, truncated, _ = self.step(act)
            if terminated or truncated:
                break
        return self.get_metrics()

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Snapshot for CRISP React GUI visualizer."""
        yard_dict: Dict[Tuple[int, int], List[int]] = {
            (1, s + 1): list(self._sim.get_stack_containers(s))
            for s in range(self._sim.num_stacks)
        }
        obs = self.get_observation()
        obs_dict: Dict[Tuple[int, int], List[float]] = {
            (1, s + 1): [obs.get_container(s, h).obs_value for h in range(obs.get_stack_height(s))]
            for s in range(self._sim.num_stacks)
        }
        recent_history = [
            {
                "src": (1, m.source.stack + 1),
                "dst": (1, m.destination.stack + 1) if not m.is_retrieval else (1, -1),
                "cid": m.container_id,
                "retrieval": m.is_retrieval,
            }
            for m in self._sim.move_history[-20:]
        ]
        return {
            "yard": yard_dict,
            "obs_yard": obs_dict,
            "metrics": self.get_metrics(),
            "history": recent_history,
        }
