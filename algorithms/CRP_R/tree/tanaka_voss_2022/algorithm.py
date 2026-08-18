"""
Tanaka & Voß (2022) improved B&B for the restricted BRP with distinct priorities.

Wraps the ``restricted-distinct-1.3`` C binary (``rbrp_bb``) which accompanies:

  S. Tanaka and S. Voß, "An exact approach to the restricted block relocation
  problem based on a new integer programming formulation,"
  European Journal of Operational Research, 296(2):485-503, 2022.

NOTE: Relationship between this module and the paper's primary algorithm
-----------------------------------------------------------------------

The primary contribution of Tanaka & Voss (2022) (Sections 3-4, Algorithm 1)
is an exact iterative algorithm based on a novel integer programming (IP)
formulation. Its key steps are:

  1. Enumerate all possible relocation sequences for each blocking block and
     formulate the problem as a binary IP (the "IP formulation").
  2. Build a relaxed formulation (RP) using truncated relocation sequences
     as a lower bound.
  3. Derive an upper-bound formulation (UP) from (RP) by retaining only
     complete relocation sequences.
  4. Iteratively expand truncated sequences, update (RP)/(UP), and repeat
     until the optimality gap reaches zero.
  5. Solve each (RP)/(UP) instance with Gurobi Optimizer (MIP solver).

This module instead wraps the *comparison* B&B algorithm described in
Section 5.2 of the same paper -- an improved version of the branch-and-bound
from Tanaka & Mizuno (2018), with the following enhancements introduced in
the 2022 paper:

  - Tighter lower bounds: LB-LIS (Quispe et al. 2018) and UBALB
    (Bacci et al. 2019) replace LB4, with early termination as soon as
    the lower bound exceeds the current upper bound.
  - OpenMP multi-threading support (-m flag).
  - Backtrack-and-restart search strategy (-b flag).

Vendor C source: vendor/restricted-distinct-1.3/
  - solve.c      : B&B main search (functions bb / bb_sub).
  - lower_bound.c: Lower bound implementations; LOWER_BOUND=5 (LB-LIS) active.
  - heuristics.c : Greedy heuristic embedded in B&B for upper-bound updates.

The paper's primary IP-based algorithm (Algorithm 1) is NOT yet integrated
into this platform. It requires a C++ implementation of the IP modelling /
sequence expansion logic plus Gurobi (or a compatible MIP solver).
The authors' original C++ source is available at:
  https://sites.google.com/site/shunjitanaka/brp

Key improvements in this B&B over the 2018 solver (``restricted-distinct-1.11``):
  * Optional OpenMP multi-threading (``-m`` flag).
  * Optional backtrack-and-restart search strategy (``-b`` flag).
  * Tighter lower bounds: LB-LIS (Quispe et al. 2018) and UBALB (Bacci et al. 2019),
    with early termination once the lower bound exceeds the current upper bound.

The input/output format is identical to the 2018 solver, so the same
``tanaka_export`` helper is reused.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.layout_trace import trace_layout

from .tanaka_export import yard_to_tanaka_instance


class TanakaBB2022(BaseAlgorithm):
    # "BB2022" in the class name distinguishes this comparison B&B from the
    # paper's primary IP-based algorithm, which is not implemented here.
    name = "Tanaka & Voß (2022) B&B"
    category = "Exact"
    description = (
        "Improved branch-and-bound for restricted BRP with distinct priorities "
        "(Tanaka & Voss 2022, EJOR 296:485-503). "
        "Note: this is the comparison B&B algorithm from the paper, not the "
        "primary IP-based iterative algorithm. "
        "Supports OpenMP multi-threading and optional backtrack-restart. "
        "Uses ``restricted-distinct-1.3`` (``rbrp_bb``). "
        "Primary metric: relocation count."
    )
    compatible_problems = ["CRP-R"]
    # Parse opt=N (proven optimal) and best=N (best found, optimality not proven)
    # from rbrp_bb stderr output.
    _OPT_RE  = re.compile(r"opt=(\d+)")
    _BEST_RE = re.compile(r"best=(\d+)")

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    def _default_binary(self) -> Path:
        return (
            Path(__file__).resolve().parent
            / "vendor"
            / "restricted-distinct-1.3"
            / "rbrp_bb"
        )

    def _resolve_binary(self) -> Path:
        extra = self.config.extra or {}
        override = (
            extra.get("tanaka2022_binary")
            or os.environ.get("TANAKA2022_RBRP_BB")
        )
        if override:
            return Path(str(override)).expanduser().resolve()
        return self._default_binary()

    def _run_bb(
        self,
        instance_text: str,
        max_tiers: int,
        time_limit_sec: int,
        n_threads: int,
        backtrack: bool,
    ) -> Tuple[int, bool, str]:
        binary = self._resolve_binary()
        if not binary.is_file():
            raise FileNotFoundError(
                f"Tanaka & Voß 2022 solver binary not found at {binary}. "
                "Run `make` in vendor/restricted-distinct-1.3 or set "
                "`tanaka2022_binary` / env `TANAKA2022_RBRP_BB`."
            )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tanaka", delete=False, encoding="utf-8"
        ) as tf:
            tf.write(instance_text)
            tmp_path = tf.name

        try:
            cmd = [
                str(binary),
                "-s",
                "-T", str(int(max_tiers)),
                "-t", str(int(time_limit_sec)),
                "-m", str(int(n_threads)),
            ]
            if backtrack:
                cmd.append("-b")
            cmd.append(tmp_path)

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=float(time_limit_sec) + 60.0,
                check=False,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        blob = (proc.stderr or "") + "\n" + (proc.stdout or "")
        m_opt  = self._OPT_RE.search(blob)
        m_best = self._BEST_RE.search(blob)
        if m_opt:
            return int(m_opt.group(1)), True, blob[-4000:]
        if m_best:
            return int(m_best.group(1)), False, blob[-4000:]
        raise RuntimeError(
            "Tanaka & Voß 2022 rbrp_bb produced no opt=/best= line.\n"
            f"returncode={proc.returncode}\n---\n{blob[-8000:]}"
        )

    def train(
        self,
        problem_factory: Callable,
        result_queue: mp.Queue,
        stop_event: mp.Event,
    ) -> None:
        cfg_a   = self.config
        n_seeds = 1  # multi-seed eval removed; single run only
        extra   = cfg_a.extra or {}

        t_limit   = int(extra.get("tanaka2022_time_limit_sec", 600))
        n_threads = int(extra.get("tanaka2022_n_threads", 1))
        backtrack = bool(extra.get("tanaka2022_backtrack", False))

        all_metrics: List[Dict[str, float]] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            trace_layout(
                f"TanakaBB2022.train: seed {seed + 1}/{n_seeds}  "
                f"threads={n_threads}  backtrack={backtrack}"
            )

            env = problem_factory()
            env.reset(seed=seed, options={"skip_auto_retrieve": True})

            text = yard_to_tanaka_instance(env.config, env.yard)
            n_reloc, optimal, _dbg = self._run_bb(
                text,
                max_tiers=env.config.max_tiers,
                time_limit_sec=t_limit,
                n_threads=n_threads,
                backtrack=backtrack,
            )

            metrics = {
                "relocations":    float(n_reloc),
                "steps":          float(n_reloc),
                "time":           0.0,
                "optimal_proven": 1.0 if optimal else 0.0,
                "tanaka2022_time_limit_sec": float(t_limit),
            }
            all_metrics.append(metrics)
            primary = float(n_reloc)

            if primary < self._best_metric:
                self._best_solution = []

            self._push(
                result_queue,
                step     = seed + 1,
                metric   = primary,
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
                step=n_seeds,
                metric=self._best_metric,
                metrics=agg,
                progress=1.0,
            )

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    @classmethod
    def config_schema(cls) -> Dict[str, Any]:
        base = super().config_schema()
        base.update({
            "tanaka2022_time_limit_sec": {
                "type": "int", "default": 600, "min": 1, "max": 86400,
                "label": "Time limit (s)",
                "help": "Passed to rbrp_bb ``-t``. Hard wall-clock limit per run.",
            },
            "tanaka2022_n_threads": {
                "type": "int", "default": 1, "min": 0, "max": 64,
                "label": "OpenMP threads (0 = auto)",
                "help": (
                    "Passed to rbrp_bb ``-m``. "
                    "0 lets the solver choose the number of threads automatically."
                ),
            },
            "tanaka2022_backtrack": {
                "type": "bool", "default": False,
                "label": "Backtrack-restart (-b)",
                "help": (
                    "Enable the backtrack-and-restart search strategy. "
                    "Can improve performance on hard instances at the cost "
                    "of higher memory usage."
                ),
            },
            "tanaka2022_binary": {
                "type": "str", "default": "",
                "label": "Path to rbrp_bb (optional)",
                "help": (
                    "Leave empty to use bundled vendor/restricted-distinct-1.3/rbrp_bb. "
                    "Can also be set via env var TANAKA2022_RBRP_BB."
                ),
            },
        })
        return base
