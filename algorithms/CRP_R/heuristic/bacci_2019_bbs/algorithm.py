"""
Bacci, Mattia, Ventura (2019) — Bounded Beam Search (BBS) heuristic for CRP-R.

Algorithm overview
------------------
The BBS heuristic uses a beam search with adaptive beam width β (dependent on
problem size) to explore relocation sequences.  At each node, three fast inner
heuristics compute upper bounds used to guide and prune the search:

  • ChainF    — Jovanović & Voß (2014) chain heuristic (used when n < 1 000)
  • Difference1 — Ünlüyurt & Aydin (2012) (used when 1 000 ≤ n < 10 000)
  • GAH       — Wu & Ting (2012) (used when n ≥ 10 000)

Lower bounds during search come from the UBALB bound (Bacci et al., 2019).

Beam widths (hardcoded, size-adaptive):
  n <  40  → β = 800     n < 60  → β = 500
  n < 80   → β = 300     n < 100 → β = 200
  n < 120  → β = 100     n ≥ 120 → β = 50

The algorithm terminates either when the search tree is exhausted or when the
configurable time limit is reached; it always returns the best feasible
solution found so far.

Integration notes
-----------------
The C++ source (`rBRP_BSheu.cpp`) is compiled once into `vendor/bbs_heuristic`.
Each `train()` call converts the current platform yard to the BC-RBRP input
format, runs the binary via subprocess, and parses the relocation count from
stdout.  No step-by-step environment interaction is needed: BBS solves the
full instance in one shot.

Config parameters
-----------------
  num_eval_seeds   : int  — seeds (layouts) to evaluate (default 10)
  bbs_time_limit   : float — per-instance time limit in seconds (default 5.0)
  bbs_binary       : str  — override path to compiled binary (optional)

Compatible problems
-------------------
  CRP-R

Reference
---------
T. Bacci, S. Mattia, P. Ventura,
"The bounded beam search algorithm for the block relocation problem",
Computers & Operations Research, 103, 252–264, 2019.
https://doi.org/10.1016/j.cor.2018.11.011
"""

from __future__ import annotations

import multiprocessing as mp
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.layout_trace import trace_layout
from .bbs_export import yard_to_bbs_instance


class BacciBBS(BaseAlgorithm):

    name                = "Bacci et al. (2019) BBS"
    category            = "Heuristic"
    description         = (
        "[single-bay origin]  "
        "Bacci, Mattia, Ventura (Computers & OR 2019, 103:252–264) Bounded Beam Search for CRP-R. "
        "Adaptive beam width β (800 → 50 depending on n); guided by ChainF / "
        "Difference1 / GAH upper bounds and UBALB lower bound. "
        "Returns best feasible solution within the time limit."
    )
    compatible_problems = ["CRP-R"]
    step_label          = "Seed"

    _RESHUFFLES_RE = re.compile(r"reshuffles=(\d+)")

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    # ---------------------------------------------------------------- #
    # Binary resolution                                                  #
    # ---------------------------------------------------------------- #

    def _default_binary(self) -> Path:
        return Path(__file__).resolve().parent / "vendor" / "bbs_heuristic"

    def _resolve_binary(self) -> Path:
        override = self.config.extra.get("bbs_binary", "")
        if override:
            p = Path(override)
            if p.is_file():
                return p
        p = self._default_binary()
        if not p.is_file():
            raise FileNotFoundError(
                f"BBS binary not found at {p}. "
                "Re-compile with:\n"
                "  cd vendor && gcc -O3 -Wno-unused-result -x c++ "
                "-o bbs_heuristic bbs_main.cpp rBRP_BSheu.cpp -lm"
            )
        return p

    # ---------------------------------------------------------------- #
    # Core subprocess call                                               #
    # ---------------------------------------------------------------- #

    def _run_bbs(
        self,
        instance_text:  str,
        time_limit_sec: float,
    ) -> Tuple[int, str]:
        """
        Write instance to a temp file, run the BBS binary, and parse output.

        Returns
        -------
        (relocations, raw_stdout)
        """
        binary = self._resolve_binary()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as tmp:
            tmp.write(instance_text)
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                [str(binary), tmp_path, str(time_limit_sec)],
                capture_output=True,
                text=True,
                timeout=time_limit_sec + 30,
            )
            stdout = result.stdout.strip()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        m = self._RESHUFFLES_RE.search(stdout)
        if m:
            return int(m.group(1)), stdout
        raise RuntimeError(
            f"BBS binary produced unexpected output: {stdout!r}\n"
            f"stderr: {result.stderr[:300]!r}"
        )

    # ---------------------------------------------------------------- #
    # Training loop                                                      #
    # ---------------------------------------------------------------- #

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        cfg            = self.config
        n_seeds        = max(1, cfg.num_eval_seeds)
        time_limit     = float(cfg.extra.get("bbs_time_limit", 5.0))
        all_metrics: List[Dict] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            trace_layout(
                f"BacciBBS.train: seed {seed + 1}/{n_seeds}  "
                f"time_limit={time_limit}s"
            )

            env = problem_factory()
            env.config.seed = seed
            env.reset(options={"skip_auto_retrieve": True})

            instance_text = yard_to_bbs_instance(env.config, env.yard)
            relocations, _ = self._run_bbs(instance_text, time_limit)

            metrics = {
                "relocations": float(relocations),
                "steps":       float(relocations),
                "time":        float(relocations),
                "progress":    1.0,
            }
            all_metrics.append(metrics)

            if relocations < self._best_metric:
                self._best_metric   = float(relocations)
                self._best_solution = []

            self._push(
                result_queue,
                step     = seed + 1,
                metric   = float(relocations),
                metrics  = metrics,
                progress = (seed + 1) / n_seeds,
                snapshot = env.get_state_snapshot(),
            )

        if all_metrics:
            agg = {
                k: float(np.mean([m[k] for m in all_metrics if k in m]))
                for k in all_metrics[0]
            }
            self._push(
                result_queue,
                step     = n_seeds,
                metric   = self._best_metric,
                metrics  = agg,
                progress = 1.0,
            )

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    # ---------------------------------------------------------------- #
    # Configuration schema                                               #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "num_eval_seeds": {
                "type":    "int",
                "default": 10,
                "min":     1,
                "max":     200,
                "label":   "Evaluation seeds",
                "help":    "Number of random initial layouts to evaluate.",
            },
            "bbs_time_limit": {
                "type":    "float",
                "default": 5.0,
                "min":     0.1,
                "max":     300.0,
                "label":   "Time limit per instance (s)",
                "help": (
                    "Per-instance wall-clock time limit for the BBS search. "
                    "The algorithm returns the best solution found within this budget. "
                    "For n < 40 (Zhu 5-8-39 instances) the search typically "
                    "finishes in well under 1 s."
                ),
            },
            "bbs_binary": {
                "type":    "str",
                "default": "",
                "label":   "Binary path override (optional)",
                "help": (
                    "Leave empty to use the pre-compiled binary in vendor/. "
                    "Set an absolute path to use a custom build."
                ),
            },
        })
        return base
