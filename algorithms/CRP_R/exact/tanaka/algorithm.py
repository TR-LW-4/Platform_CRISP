"""
Tanaka ``restricted-duplicate-1.01`` branch-and-bound (exact BRP solver).

Bridges the vendor C binary ``brp_bb`` to :class:`core.base_algorithm.BaseAlgorithm`.
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


class TanakaBB(BaseAlgorithm):
    """
    Shunji Tanaka — restricted relocation branch-and-bound (CRP-R naming).

    Vendor code lives under ``vendor/restricted-duplicate-1.01`` (BSD licence).
    """

    name                = "Tanaka (2016) B&B"
    category            = "Exact"
    description         = (
        "Exact branch-and-bound for restricted BRP (Tanaka ``restricted-duplicate-1.01``). "
        "Exports the initial yard to Tanaka format and runs ``brp_bb``. "
        "Primary metric = relocation count; stderr reports ``opt=`` (proven optimal) "
        "or ``best=`` (time limit / interrupted)."
    )
    compatible_problems = ["CRP-R"]
    step_label          = "Seed"

    _OPT_RE = re.compile(r"opt=(\d+)")
    _BEST_RE = re.compile(r"best=(\d+)")

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    def _default_binary(self) -> Path:
        return (
            Path(__file__).resolve().parent
            / "vendor"
            / "restricted-duplicate-1.01"
            / "brp_bb"
        )

    def _resolve_binary(self) -> Path:
        extra = self.config.extra or {}
        override = extra.get("tanaka_binary") or os.environ.get("TANAKA_BRP_BB")
        if override:
            return Path(str(override)).expanduser().resolve()
        return self._default_binary()

    def _run_bb(
        self,
        instance_text: str,
        max_tiers: int,
        time_limit_sec: int,
    ) -> Tuple[int, bool, str]:
        """
        Returns (relocations, optimal_proven, combined_stderr_stdout_tail).
        """
        binary = self._resolve_binary()
        if not binary.is_file():
            raise FileNotFoundError(
                f"Tanaka solver binary not found at {binary}. "
                "Run `make` in vendor/restricted-duplicate-1.01 or set "
                "`tanaka_binary` / env `TANAKA_BRP_BB`."
            )

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".tanaka",
            delete=False,
            encoding="utf-8",
        ) as tf:
            tf.write(instance_text)
            tmp_path = tf.name

        try:
            cmd = [
                str(binary),
                "-s",
                "-T",
                str(int(max_tiers)),
                "-t",
                str(int(time_limit_sec)),
                tmp_path,
            ]
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
        m_opt = self._OPT_RE.search(blob)
        m_best = self._BEST_RE.search(blob)
        if m_opt:
            return int(m_opt.group(1)), True, blob[-4000:]
        if m_best:
            return int(m_best.group(1)), False, blob[-4000:]
        raise RuntimeError(
            "Tanaka brp_bb produced no opt=/best= line.\n"
            f"returncode={proc.returncode}\n---\n{blob[-8000:]}"
        )

    def train(
        self,
        problem_factory: Callable,
        result_queue: mp.Queue,
        stop_event: mp.Event,
    ) -> None:
        cfg_a       = self.config
        n_seeds     = max(1, cfg_a.num_eval_seeds)
        extra       = cfg_a.extra or {}
        t_limit     = int(extra.get("tanaka_time_limit_sec", 600))
        all_metrics: List[Dict[str, float]] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            trace_layout(
                f"TanakaBB.train: seed {seed + 1}/{n_seeds}"
            )

            env = problem_factory()
            env.config.seed = seed
            env.reset(seed=seed, options={"skip_auto_retrieve": True})

            text = yard_to_tanaka_instance(env.config, env.yard)
            n_reloc, optimal, _dbg = self._run_bb(
                text,
                max_tiers=env.config.max_tiers,
                time_limit_sec=t_limit,
            )

            metrics = {
                "relocations":     float(n_reloc),
                "steps":           float(n_reloc),
                "time":            0.0,
                "optimal_proven":  1.0 if optimal else 0.0,
                "tanaka_time_limit_sec": float(t_limit),
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
                step     = n_seeds,
                metric   = self._best_metric,
                metrics  = agg,
                progress = 1.0,
            )

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    @classmethod
    def config_schema(cls) -> Dict[str, Any]:
        base = super().config_schema()
        base.update({
            "num_eval_seeds": {
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 100,
                "label": "Evaluation seeds",
                "help": "Number of independent evaluations (same layout if layout-driven).",
            },
            "tanaka_time_limit_sec": {
                "type": "int",
                "default": 600,
                "min": 1,
                "max": 86400,
                "label": "Tanaka B&B time limit (s)",
                "help": "Passed to brp_bb ``-t`` (CPU-time style limit inside solver).",
            },
            "tanaka_binary": {
                "type": "str",
                "default": "",
                "label": "Path to brp_bb (optional)",
                "help": (
                    "Leave empty to use bundled vendor/brp_bb. "
                    "Override or set env TANAKA_BRP_BB if you compile elsewhere."
                ),
            },
        })
        return base
