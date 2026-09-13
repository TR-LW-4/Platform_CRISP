"""
TanakaBB2022
<2022> <exact> <restricted> <single-bay> <CRP-R>
Improved branch-and-bound wrapper (restricted-distinct-1.3)
tanaka2022_time_limit_sec --- 600 --- Per-instance time limit (s)
tanaka2022_n_threads --- 1 --- OpenMP threads (0 = auto)

------------------------------- Reference --------------------------------
S. Tanaka, S. Voß,
"An exact approach to the restricted block relocation problem based on
 a new integer programming formulation",
European Journal of Operational Research 296 (2022) 485–503.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
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

    name = "Tanaka & Voß (2022) B&B"
    category = "Exact"
    description = "Tanaka & Voß (EJOR 2022) improved branch-and-bound."
    compatible_problems = ["CRP-R"]
    geometry = "single-bay"
    objectives = ["relocations"]
    fidelity = "faithful"
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
