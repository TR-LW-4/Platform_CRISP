"""
Abstract base class for all algorithms (RL and Evolutionary).

Design
------
- train() runs in a *subprocess* (via multiprocessing) so the GUI stays responsive.
- Every algorithm sends progress updates through a multiprocessing.Queue.
- The GUI reads the queue at a fixed interval and refreshes plots.
- stop_event.set() from the GUI asks the algorithm to terminate gracefully.
"""

from __future__ import annotations

import multiprocessing as mp
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ================================================================ #
#  Algorithm configuration                                           #
# ================================================================ #

@dataclass
class AlgorithmConfig:
    """
    Universal configuration shared by all algorithms.
    Algorithm subclasses may add their own fields via `extra`.
    """

    # ── Common ───────────────────────────────────────────────────── #
    max_iterations: int   = 500
    seed:           int   = 0
    report_interval: int  = 10    # push to queue every N iterations
    num_eval_seeds:  int  = 5     # seeds used for evaluation (EA)

    # ── RL-specific ───────────────────────────────────────────────── #
    total_timesteps: int   = 200_000
    learning_rate:   float = 3e-4
    gamma:           float = 0.99
    num_envs:        int   = 4
    num_steps:       int   = 256   # steps per rollout
    batch_size:      int   = 64
    hidden_dim:      int   = 128

    # ── Evolutionary-specific ─────────────────────────────────────── #
    population_size:   int   = 50
    crossover_rate:    float = 0.8
    mutation_rate:     float = 0.1
    tournament_size:   int   = 3
    elite_count:       int   = 2

    # ── Extra (algorithm-specific overrides) ──────────────────────── #
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "extra"}
        d.update(self.extra)
        return d


# ================================================================ #
#  Progress record (what algorithms push to the queue)              #
# ================================================================ #

@dataclass
class ProgressRecord:
    """Single progress update sent from training subprocess to GUI."""
    step:          int
    metric:        float              # primary metric (lower is better)
    metrics:       Dict[str, float]   # all metrics
    best_metric:   float
    progress:      float              # 0.0 – 1.0
    yard_snapshot: Optional[Dict]     = None
    extra:         Dict[str, Any]     = field(default_factory=dict)


# ================================================================ #
#  Base algorithm                                                    #
# ================================================================ #

class BaseAlgorithm(ABC):
    """
    Abstract base for every algorithm.

    Subclasses must implement
    ─────────────────────────
    train(problem_factory, result_queue, stop_event)
    get_best_solution()

    Class-level metadata (override in subclass)
    ────────────────────────────────────────────
    name                : str  – display name in GUI
    category            : str  – "RL" | "Evolutionary" | "Heuristic"
    description         : str
    compatible_problems : list – problem class *names*; empty = all problems
    """

    # ── Metadata (override in subclasses) ─────────────────────────── #
    name:                str       = "BaseAlgorithm"
    category:            str       = "Unknown"
    description:         str       = ""
    compatible_problems: List[str] = []   # empty → compatible with all

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        self.config = config if config is not None else AlgorithmConfig()
        self._best_solution: Optional[List[int]] = None
        self._best_metric:   float               = float("inf")

    # ---------------------------------------------------------------- #
    # Abstract interface                                                  #
    # ---------------------------------------------------------------- #

    @abstractmethod
    def train(
        self,
        problem_factory: Callable,   # () → BaseProblem instance
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        """
        Main training loop.  *Runs in a subprocess.*

        Parameters
        ----------
        problem_factory : zero-argument callable that returns a fresh problem.
        result_queue    : put ProgressRecord (or plain dict) here periodically.
        stop_event      : poll stop_event.is_set() to support user-requested stop.

        Implementations should:
          - Call problem_factory() to get a fresh environment.
          - Push a ProgressRecord every self.config.report_interval iterations.
          - Respect stop_event for clean shutdown.
        """

    @abstractmethod
    def get_best_solution(self) -> Optional[List[int]]:
        """Return the best solution (action sequence) found so far."""

    # ---------------------------------------------------------------- #
    # Convenience helpers                                                 #
    # ---------------------------------------------------------------- #

    def _push(
        self,
        q:            mp.Queue,
        step:         int,
        metric:       float,
        metrics:      Dict[str, float],
        progress:     float,
        snapshot:     Optional[Dict] = None,
        extra:        Optional[Dict] = None,
    ) -> None:
        """Helper: push a ProgressRecord, never raises (GUI queue may be full)."""
        if metric < self._best_metric:
            self._best_metric = metric
        record = ProgressRecord(
            step=step,
            metric=metric,
            metrics=metrics,
            best_metric=self._best_metric,
            progress=min(progress, 1.0),
            yard_snapshot=snapshot,
            extra=extra or {},
        )
        try:
            q.put_nowait(record)
        except Exception:
            pass   # queue full – skip this update

    # ---------------------------------------------------------------- #
    # Config schema for GUI                                              #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict[str, Dict]:
        """
        Return a schema used by the GUI to build parameter widgets.
        Override to expose algorithm-specific parameters.
        """
        return {
            "max_iterations":  {"type": "int",   "default": 500,    "min": 10,    "max": 100_000},
            "seed":            {"type": "int",   "default": 0,      "min": 0,     "max": 9999},
            "report_interval": {"type": "int",   "default": 10,     "min": 1,     "max": 100},
        }


# ================================================================ #
#  Subprocess launcher (used by GUI)                                 #
# ================================================================ #

class TrainingSession:
    """
    Manages the lifecycle of an algorithm training subprocess.

    Usage (from GUI)
    ----------------
    session = TrainingSession(algo, problem_factory)
    session.start()
    # poll:
    records = session.drain()
    # stop:
    session.stop()
    """

    def __init__(
        self,
        algorithm:       BaseAlgorithm,
        problem_factory: Callable,
        queue_maxsize:   int = 512,
    ):
        self.algorithm       = algorithm
        self.problem_factory = problem_factory
        self._queue          = mp.Queue(maxsize=queue_maxsize)
        self._stop_event     = mp.Event()
        self._process: Optional[mp.Process] = None

    def start(self) -> None:
        self._process = mp.Process(
            target=self.algorithm.train,
            args=(self.problem_factory, self._queue, self._stop_event),
            daemon=True,
        )
        self._process.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._process and self._process.is_alive():
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()

    def is_running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def drain(self) -> List[ProgressRecord]:
        """Return all available records without blocking."""
        records = []
        while True:
            try:
                records.append(self._queue.get_nowait())
            except Exception:
                break
        return records
