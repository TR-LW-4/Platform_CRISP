"""
Caserta, Voß, Sniedovich (2011) — Corridor Method (CM) for CRP-R.

Algorithm overview
------------------
A DP-inspired metaheuristic that restricts the search space at each step by
imposing a two-dimensional "corridor" around the current bay configuration:

  Horizontal corridor (δ):  block r in stack i can only be relocated to
    stacks within the range [i−δ, i+δ].  Smaller δ = faster but fewer
    options explored per step.

  Vertical corridor (λ):  sets the maximum height of a target stack.
    The original paper uses λ = H + 2 (constant vertical corridor, -c 1).
    This is fixed to H_max + 2 in the binary and passed via -n.

Within each corridor, the algorithm evaluates the goodness of each feasible
move using a greedy score (Kim–Hong / Min–Max heuristic applied until the bay
is cleared from the proposed new state).  The move with the best score is
selected (stochastically via roulette-wheel — not purely deterministic).
The process repeats until all containers are retrieved.  Multiple trajectories
are run until the time limit is reached.

Why the CM is an important baseline
-------------------------------------
Almost every paper on CRP-R benchmarks against CM.  It is the most widely
cited comparison algorithm in the BRP/CRP literature and is significantly
better than simple greedy heuristics (e.g., Kim–Hong) on large instances
(Table 3 of the paper, 10×10: CM = 128.3 vs KH = 178.6 relocations).

Config parameters
-----------------
  delta          : int   — horizontal corridor half-width (default 2)
  time_limit     : float — per-instance wall-clock limit in seconds (default 60)
  cm_binary      : str   — override path to compiled binary (optional)
  num_eval_seeds : int   — 1 (CM is stochastic but time-limited; 1 seed/layout)

Compatible problems
-------------------
  CRP-R

Reference
---------
M. Caserta, S. Voß, M. Sniedovich,
"Applying the corridor method to a blocks relocation problem",
OR Spectrum, 33, 915–929, 2011.
https://doi.org/10.1007/s00291-009-0176-5

Note: The C++ source (vendor/) is the 2017-revised version of the OR Spectrum
code by Marco Caserta (marco.caserta@ie.edu), which fixes a height-limit bug
present in the original 2009 implementation.
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
from .cm_export import yard_to_cm_instance


class CasertaCM(BaseAlgorithm):

    name                = "Caserta et al. (2011) CM"
    category            = "Heuristic"
    description         = (
        "[single-bay origin]  "
        "Caserta, Voß, Sniedovich (OR Spectrum 2011) Corridor Method for CRP-R. "
        "DP-inspired metaheuristic: at each step, restricts feasible destinations "
        "to a horizontal corridor [i±δ] and a vertical corridor (max height H+2). "
        "Best greedy-scored move selected via roulette-wheel; multi-restart until "
        "time limit. Standard baseline in BRP literature — outperforms Kim–Hong "
        "by ~28% on 10×10 instances."
    )
    compatible_problems = ["CRP-R"]
    step_label          = "Seed"

    _MOVES_RE = re.compile(r"CM\s*:\s*Solution found with\s*(\d+)\s*moves")

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    # ---------------------------------------------------------------- #
    # Binary resolution                                                  #
    # ---------------------------------------------------------------- #

    def _default_binary(self) -> Path:
        return Path(__file__).resolve().parent / "vendor" / "brp_cm"

    def _resolve_binary(self) -> Path:
        override = self.config.extra.get("cm_binary", "")
        if override:
            p = Path(override)
            if p.is_file():
                return p
        p = self._default_binary()
        if not p.is_file():
            raise FileNotFoundError(
                f"CM binary not found at {p}. "
                "Re-compile with:\n"
                "  cd vendor && CPLUS_INCLUDE_PATH='' g++ -O3 -DREPL "
                "timer.cpp options.cpp heuristic.cpp containers.cpp -o brp_cm"
            )
        return p

    # ---------------------------------------------------------------- #
    # Core subprocess call                                               #
    # ---------------------------------------------------------------- #

    def _run_cm(
        self,
        instance_text: str,
        delta:         int,
        max_height:    int,
        time_limit:    float,
    ) -> Tuple[int, str]:
        """
        Write instance to a temp file, run the CM binary, and parse output.

        Parameters
        ----------
        delta      : horizontal corridor half-width (-d parameter)
        max_height : maximum stack height (-n parameter; paper uses H+2)
        time_limit : wall-clock time limit in seconds (-t parameter)

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
                [
                    str(binary),
                    "-f", tmp_path,
                    "-d", str(delta),
                    "-n", str(max_height),
                    "-t", str(int(time_limit)),
                    "-c", "1",   # constant vertical corridor (H+2 style)
                ],
                capture_output=True,
                text=True,
                timeout=time_limit + 30,
                env={"CPLUS_INCLUDE_PATH": ""},  # prevent sandbox include-path interference
            )
            stdout = result.stdout.strip()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        m = self._MOVES_RE.search(stdout)
        if m:
            return int(m.group(1)), stdout
        raise RuntimeError(
            f"CM binary produced unexpected output: {stdout!r}\n"
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
        cfg        = self.config
        n_seeds    = max(1, cfg.num_eval_seeds)
        delta      = int(cfg.extra.get("delta", 2))
        time_limit = float(cfg.extra.get("time_limit", 60.0))
        all_metrics: List[Dict] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            trace_layout(
                f"CasertaCM.train: seed {seed+1}/{n_seeds}  "
                f"delta={delta}  time_limit={time_limit}s"
            )

            env = problem_factory()
            env.config.seed = seed
            env.reset(options={"skip_auto_retrieve": True})

            max_height = int(env.config.max_tiers) + 2

            instance_text = yard_to_cm_instance(env.config, env.yard)
            relocations, _ = self._run_cm(instance_text, delta, max_height, time_limit)

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
                "default": 1,
                "min":     1,
                "max":     20,
                "label":   "Evaluation seeds",
                "help": (
                    "CM is stochastic — use 1 seed per fixed benchmark file. "
                    "Increase only for random-layout mode."
                ),
            },
            "delta": {
                "type":    "int",
                "default": 2,
                "min":     1,
                "max":     20,
                "label":   "Horizontal corridor half-width (δ)",
                "help": (
                    "Blocks above the target may only be relocated to stacks "
                    "within [i−δ, i+δ].  Larger δ improves quality but increases "
                    "runtime.  Paper uses δ = 1 for most small instances, δ = 2 "
                    "for larger ones (Tables 1–3)."
                ),
            },
            "time_limit": {
                "type":    "float",
                "default": 5.0,
                "min":     1.0,
                "max":     300.0,
                "label":   "Time limit per instance (s)",
                "help": (
                    "Wall-clock time limit passed to the CM binary. "
                    "The algorithm runs multiple restarts until this limit is reached "
                    "and returns the best solution found. "
                    "Default 5 s is sufficient for small/medium instances. "
                    "Paper uses 60 s for small/medium and 300 s for very large instances."
                ),
            },
            "cm_binary": {
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
