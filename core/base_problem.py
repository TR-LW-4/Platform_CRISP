"""
Abstract base class for all container relocation problems.

Design contract
---------------
Every problem:
  1. Extends gymnasium.Env  → RL algorithms can use it directly
  2. Provides evaluate(solution) → metrics dict  → EA can evaluate solutions
  3. Uses Yard as the internal state engine
  4. Declares class-level metadata (name, tags, …) for GUI auto-discovery
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from .yard import Yard
from .container import Container, make_containers


# ================================================================ #
#  Problem configuration                                             #
# ================================================================ #

@dataclass
class ProblemConfig:
    """
    Universal configuration shared by all problems.
    Problems may ignore fields that don't apply to them.

    All fields have sensible defaults so a problem can be instantiated
    with zero arguments for quick testing.
    """

    # ── Yard geometry ────────────────────────────────────────────── #
    num_bays:   int   = 4
    num_rows:   int   = 2
    max_tiers:  int   = 3

    # ── Containers ───────────────────────────────────────────────── #
    num_containers: int  = 12
    num_groups:     int  = 3
    seed:           int  = 0

    # Container attribute toggles (kept False → standard TEU, 10 t)
    enable_weight: bool = False
    enable_size:   bool = False
    enable_type:   bool = False

    # ── Crane ────────────────────────────────────────────────────── #
    num_cranes:  int   = 1
    crane_speed: float = 1.0   # "bays per time-unit"

    # ── Vessel (CSPP variants only) ───────────────────────────────── #
    vessel_bays:  int = 2
    vessel_rows:  int = 2
    vessel_tiers: int = 3

    # ── Problem-specific extras ───────────────────────────────────── #
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "extra"}
        d.update(self.extra)
        return d


# ================================================================ #
#  Base problem                                                      #
# ================================================================ #

class BaseProblem(gym.Env, ABC):
    """
    Abstract base for all problems.

    Subclasses must implement
    ─────────────────────────
    _setup_spaces()       – define observation_space, action_space
    _build_episode()      – populate yard from config (called in reset)
    reset()               – gymnasium reset
    step(action)          – gymnasium step
    evaluate(solution)    – evaluate a complete action-sequence (for EA)
    get_metrics()         – return current episode metrics dict

    Class-level metadata (override in subclass)
    ────────────────────────────────────────────
    name          : str  – display name in GUI
    description   : str  – short description
    tags          : list – e.g. ["relocation", "fixed-order"]
    metric_names  : list – ordered list of metric keys (first = primary)
    """

    # ── Metadata (override in subclasses) ─────────────────────────── #
    name:         str  = "BaseProblem"
    description:  str  = ""
    tags:         List[str] = []
    metric_names: List[str] = ["relocations"]

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        config: Optional[ProblemConfig] = None,
        render_mode: Optional[str]      = None,
    ):
        super().__init__()
        self.config      = config if config is not None else ProblemConfig()
        self.render_mode = render_mode

        # Build yard (dimensions only – containers added in reset)
        self.yard = Yard(
            self.config.num_bays,
            self.config.num_rows,
            self.config.max_tiers,
        )

        # Container list is populated by _build_episode()
        self.containers: List[Container] = []

        self._setup_spaces()

    # ---------------------------------------------------------------- #
    # Abstract interface                                                  #
    # ---------------------------------------------------------------- #

    @abstractmethod
    def _setup_spaces(self) -> None:
        """Define self.observation_space and self.action_space."""

    @abstractmethod
    def _build_episode(self) -> None:
        """
        Populate self.yard with containers for a new episode.
        Called inside reset() after yard.clear().
        """

    @abstractmethod
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """gymnasium reset → (obs, info)."""

    @abstractmethod
    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """gymnasium step → (obs, reward, terminated, truncated, info)."""

    @abstractmethod
    def evaluate(self, solution: List[int]) -> Dict[str, float]:
        """
        Evaluate a *complete* action sequence (list of integer actions).

        For evolutionary algorithms – simulate the solution on a fresh
        episode and return all metrics.

        Returns a dict keyed by self.metric_names.
        """

    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Return a snapshot of current episode metrics."""

    # ---------------------------------------------------------------- #
    # Shared helpers                                                      #
    # ---------------------------------------------------------------- #

    def _make_containers(self) -> List[Container]:
        """Create containers according to config."""
        return make_containers(
            n=self.config.num_containers,
            num_groups=self.config.num_groups,
            seed=self.config.seed,
            enable_weight=self.config.enable_weight,
            enable_size=self.config.enable_size,
            enable_type=self.config.enable_type,
        )

    def get_state_snapshot(self) -> Dict:
        """
        Full snapshot for GUI visualisation.
        Returns yard state + current metrics + move history.
        """
        return {
            "yard":    self.yard.group_snapshot(),
            "metrics": self.get_metrics(),
            "history": list(self.yard.move_history[-20:]),   # last 20 moves
        }

    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """Default: delegate to visualisation module if available."""
        try:
            from visualization.bay_renderer import render_yard
            return render_yard(self.yard, self.config)
        except ImportError:
            return None

    # ---------------------------------------------------------------- #
    # Config schema for GUI                                              #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict[str, Dict]:
        """
        Return a schema dict used by the GUI to build parameter widgets.
        Each key → {"type": "int"|"float"|"bool", "default": …, "min": …, "max": …}
        """
        return {
            "num_bays":       {"type": "int",   "default": 4,  "min": 1, "max": 20},
            "num_rows":       {"type": "int",   "default": 2,  "min": 1, "max": 10},
            "max_tiers":      {"type": "int",   "default": 3,  "min": 2, "max": 10},
            "num_containers": {"type": "int",   "default": 12, "min": 1, "max": 200},
            "num_groups":     {"type": "int",   "default": 3,  "min": 1, "max": 20},
            "num_cranes":     {"type": "int",   "default": 1,  "min": 1, "max": 4},
            "crane_speed":    {"type": "float", "default": 1.0,"min": 0.1, "max": 5.0},
            "enable_weight":  {"type": "bool",  "default": False},
            "enable_size":    {"type": "bool",  "default": False},
            "enable_type":    {"type": "bool",  "default": False},
            "seed":           {"type": "int",   "default": 0,  "min": 0, "max": 9999},
        }
