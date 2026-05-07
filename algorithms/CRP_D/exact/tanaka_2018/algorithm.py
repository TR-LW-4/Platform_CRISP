"""
Tanaka ``restricted-distinct-1.11`` adapter under CRP-D (2018 revision).

Same backend as CRP-R 2018 exact adapter; metric keys are mapped to CRP-D
(shifters-first) dashboards.
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


class Tanaka2018BBForCRPD(BaseAlgorithm):
    name                = "Tanaka (2018) B&B [CRP-D]"
    category            = "Exact"
    description         = (
        "Exact B&B via Tanaka ``restricted-distinct-1.11`` (2018 revision), "
        "exposed under CRP-D with shifters-oriented metrics."
    )
    compatible_problems = ["CRP-D"]
    step_label          = "Seed"

    _OPT_RE = re.compile(r"opt=(\d+)")
    _BEST_RE = re.compile(r"best=(\d+)")

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    def _default_binary(self) -> Path:
        return (
            Path(__file__).resolve().parents[3]
            / "CRP_R"
            / "exact"
            / "tanaka_2018"
            / "vendor"
            / "restricted-distinct-1.11"
            / "brp_bb"
        )

    def _resolve_binary(self) -> Path:
        extra = self.config.extra or {}
        override = extra.get("tanaka2018_binary") or os.environ.get("TANAKA2018_BRP_BB")
        if override:
            return Path(str(override)).expanduser().resolve()
        return self._default_binary()

    def _run_bb(self, instance_text: str, max_tiers: int, time_limit_sec: int) -> Tuple[int, bool, str]:
        binary = self._resolve_binary()
        if not binary.is_file():
            raise FileNotFoundError(
                f"Tanaka 2018 solver binary not found at {binary}. "
                "Build CRP_R exact tanaka_2018 vendor binary or set override path."
            )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tanaka", delete=False, encoding="utf-8") as tf:
            tf.write(instance_text)
            tmp_path = tf.name

        try:
            cmd = [str(binary), "-s", "-T", str(int(max_tiers)), "-t", str(int(time_limit_sec)), tmp_path]
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
            "Tanaka 2018 brp_bb produced no opt=/best= line.\n"
            f"returncode={proc.returncode}\n---\n{blob[-8000:]}"
        )

    def train(self, problem_factory: Callable, result_queue: mp.Queue, stop_event: mp.Event) -> None:
        cfg_a       = self.config
        n_seeds     = max(1, cfg_a.num_eval_seeds)
        extra       = cfg_a.extra or {}
        t_limit     = int(extra.get("tanaka2018_time_limit_sec", 600))
        all_metrics: List[Dict[str, float]] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            trace_layout(f"Tanaka2018BBForCRPD.train: seed {seed + 1}/{n_seeds}")

            env = problem_factory()
            env.config.seed = seed
            env.reset(seed=seed)

            text = yard_to_tanaka_instance(env.config, env.yard)
            n_reloc, optimal, _dbg = self._run_bb(text, max_tiers=env.config.max_tiers, time_limit_sec=t_limit)

            metrics = {
                "shifters":            float(n_reloc),
                "time":                float(n_reloc),
                "vessel_utilisation":  0.0,
                "relocations":         float(n_reloc),
                "optimal_proven":      1.0 if optimal else 0.0,
                "tanaka2018_time_limit_sec": float(t_limit),
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
            agg = {k: float(np.mean([m[k] for m in all_metrics if k in m])) for k in all_metrics[0]}
            self._push(result_queue, step=n_seeds, metric=self._best_metric, metrics=agg, progress=1.0)

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    @classmethod
    def config_schema(cls) -> Dict[str, Any]:
        base = super().config_schema()
        base.update({
            "num_eval_seeds": {
                "type": "int", "default": 1, "min": 1, "max": 100,
                "label": "Evaluation seeds",
                "help": "Number of independent evaluations.",
            },
            "tanaka2018_time_limit_sec": {
                "type": "int", "default": 600, "min": 1, "max": 86400,
                "label": "Tanaka 2018 B&B time limit (s)",
                "help": "Passed to brp_bb ``-t``.",
            },
            "tanaka2018_binary": {
                "type": "str", "default": "",
                "label": "Path to 2018 brp_bb (optional)",
                "help": "Optional override for vendor solver binary path.",
            },
        })
        return base
