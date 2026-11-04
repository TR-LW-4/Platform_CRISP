from __future__ import annotations

import inspect
import multiprocessing as mp
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

    
#  ProgressRecord 是 _push() 放进队列的那一条记录，里面有步数、目标值、进度和堆场快照。

# PlatEMO/Algorithms/ALGORITHM.m。


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


def trace_layout(msg: str) -> None:
    if os.environ.get("CRISP_TRACE_LAYOUT"):
        print(f"[CRISP_TRACE_LAYOUT] {msg}", file=sys.stderr, flush=True)


class BaseAlgorithm(ABC):
    
    
    
    # ── Metadata (override in subclasses) ─────────────────────────── #
    name:                str       = "BaseAlgorithm"
    category:            str       = "Unknown"
    description:         str       = ""
    compatible_problems: List[str] = []   # empty → compatible with all

    # ── Comparison metadata (vocabularies in core/algorithm_meta.py) ─ #
    # Treat these as read-only class constants; never mutate in place.
    geometry:            str       = "single-bay"
    objectives:          List[str] = ["relocations"]
    fidelity:            str       = "faithful"

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        self.config = config if config is not None else AlgorithmConfig()
        self._best_solution: Optional[List[int]] = None
        self._best_metric:   float               = float("inf")

    @abstractmethod
    def train(
        self,
        problem_factory: Callable,   # () → BaseProblem instance
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        
### PlatEMO/Algorithms/ALGORITHM.m
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

    def parameter_set(self, *defaults: Any) -> tuple:
        
        from web.backend.header_schema import parse_header

        text = inspect.getmodule(type(self)).__doc__ or ""
        names = list(parse_header(text))
        if len(defaults) != len(names):
            raise ValueError(
                f"{type(self).__name__} has {len(names)} header "
                f"parameters, got {len(defaults)}"
            )
        extra = self.config.extra or {}
        return tuple(
            extra[name] if name in extra and extra[name] != "" else default
            for name, default in zip(names, defaults)
        )

