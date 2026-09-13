"""
TanakaMizuno2018BB
<2018> <exact> <unrestricted> <single-bay> <CRP-U>
Branch-and-bound for unrestricted BRP with distinct priorities
tanaka_mizuno2018_time_limit_sec --- 600 --- Per-instance wall-clock limit (s)

------------------------------- Reference --------------------------------
S. Tanaka, F. Mizuno,
"An exact algorithm for the unrestricted block relocation problem",
Computers & Operations Research 95 (2018) 12–31.
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


class TanakaMizuno2018BB(BaseAlgorithm):
    name                = "Tanaka & Mizuno (2018) B&B"
    category            = "Exact"
    description         = (
        "Exact branch-and-bound for the Unrestricted BRP (CRP-U) with distinct "
        "priorities. Tanaka & Mizuno — Computers & Operations Research 95 (2018) "
        "12–31. Uses ``unrestricted-distinct-1.01`` (``ubrp_bb``). "
        "Primary metric: relocation count."
    )
    compatible_problems = ["CRP-U"]
    geometry            = "single-bay"
    objectives          = ["relocations"]
    fidelity            = "faithful"
    _OPT_RE  = re.compile(r"opt=(\d+)")
    _BEST_RE = re.compile(r"best=(\d+)")

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    def _default_binary(self) -> Path:
        return (
            Path(__file__).resolve().parent
            / "vendor"
            / "unrestricted-distinct-1.01"
            / "ubrp_bb"
        )

    def _resolve_binary(self) -> Path:
        extra    = self.config.extra or {}
        override = (
            extra.get("tanaka_mizuno2018_binary")
            or os.environ.get("TANAKA_MIZUNO2018_UBRP_BB")
        )
        if override:
            return Path(str(override)).expanduser().resolve()
        return self._default_binary()

    def _run_bb(
        self,
        instance_text: str,
        max_tiers: int,
        time_limit_sec: int,
    ) -> Tuple[int, bool, str]:
        binary = self._resolve_binary()
        if not binary.is_file():
            raise FileNotFoundError(
                f"Tanaka & Mizuno 2018 solver binary not found at {binary}. "
                "Run `make` in vendor/unrestricted-distinct-1.01 or set "
                "`tanaka_mizuno2018_binary` / env `TANAKA_MIZUNO2018_UBRP_BB`."
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
        m_opt  = self._OPT_RE.search(blob)
        m_best = self._BEST_RE.search(blob)
        if m_opt:
            return int(m_opt.group(1)), True, blob[-4000:]
        if m_best:
            return int(m_best.group(1)), False, blob[-4000:]
        raise RuntimeError(
            "Tanaka & Mizuno 2018 ubrp_bb produced no opt=/best= line.\n"
            f"returncode={proc.returncode}\n---\n{blob[-8000:]}"
        )

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        cfg_a   = self.config
        n_seeds = 1  # multi-seed eval removed; single run only
        extra   = cfg_a.extra or {}
        t_limit = int(extra.get("tanaka_mizuno2018_time_limit_sec", 600))

        all_metrics: List[Dict[str, float]] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            trace_layout(
                f"TanakaMizuno2018BB.train: seed {seed + 1}/{n_seeds}"
            )

            env = problem_factory()
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
                "tanaka_mizuno2018_time_limit_sec": float(t_limit),
            }
            all_metrics.append(metrics)
            primary = float(n_reloc)

            if primary < self._best_metric:
                self._best_metric   = primary
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
                step=n_seeds, metric=self._best_metric,
                metrics=agg, progress=1.0,
            )

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    @classmethod
    def config_schema(cls) -> Dict[str, Any]:
        base = super().config_schema()
        base.update({
            "tanaka_mizuno2018_time_limit_sec": {
                "type":    "int",
                "default": 600,
                "min":     1,
                "max":     86400,
                "label":   "Time limit (s)",
                "help":    "Wall-clock time limit passed to ubrp_bb ``-t``.",
            },
            "tanaka_mizuno2018_binary": {
                "type":    "str",
                "default": "",
                "label":   "Path to ubrp_bb (optional)",
                "help":    (
                    "Leave empty to use bundled "
                    "vendor/unrestricted-distinct-1.01/ubrp_bb. "
                    "Can also be set via env var TANAKA_MIZUNO2018_UBRP_BB."
                ),
            },
        })
        return base
