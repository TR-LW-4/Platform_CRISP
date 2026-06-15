"""
Jin & Tanaka (2023) UCRP-IDBB exact solver — CRP-U adapter.

Paper
-----
Bo Jin, Shunji Tanaka (2023). "An iterative deepening branch-and-bound
algorithm for the unrestricted container relocation problem."
European Journal of Operational Research.

Vendor binary
-------------
Compiled from ``vendor/`` source (``make`` produces ``vendor/main-solve``).
Override via ``extra["jin_binary"]`` or env var ``JIN_UCRP_BINARY``.

CLI interface (from solve.c)
----------------------------
    main-solve --input/-i <file> --time_limit/-t <seconds>

stdout output lines (from algorithm.c)
---------------------------------------
    [status] best_lb = X @ t / best_ub = Y @ t / time = Z / nodes = N / probe = P
    [(p: src -> dst), ...]           # final move sequence (or '?')

CRP-U specifics
---------------
- Container priorities are **distinct integers 1…N**.
- ``compatible_problems = ["CRP-U"]``.
- Primary metric = ``relocations`` (= ``best_ub`` from the solver).
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

from .jin_export import yard_to_jin_instance


class JinTanaka2023ForCRPU(BaseAlgorithm):
    """Jin & Tanaka (2023) IDBB exact solver bound to CRP-U."""

    name                = "Jin & Tanaka (2023) IDBB [CRP-U]"
    category            = "Exact"
    description         = (
        "Exact iterative-deepening branch-and-bound for the unrestricted BRP "
        "(Jin & Tanaka 2023, ucrp-idbb).  Runs the compiled ``main-solve`` "
        "binary on the current yard snapshot.  Primary metric = relocations "
        "(``best_ub``).  ``optimal_proven=1`` when ``best_lb == best_ub``."
    )
    compatible_problems = ["CRP-U"]
    step_label          = "Seed"

    _STAT_RE = re.compile(
        r"best_lb\s*=\s*(\d+).*?best_ub\s*=\s*(\d+)", re.DOTALL
    )

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    # ---------------------------------------------------------------- #
    # Binary resolution                                                  #
    # ---------------------------------------------------------------- #

    def _default_binary(self) -> Path:
        return (
            Path(__file__).resolve().parent / "vendor" / "main-solve"
        )

    def _resolve_binary(self) -> Path:
        extra    = self.config.extra or {}
        override = extra.get("jin_binary") or os.environ.get("JIN_UCRP_BINARY")
        if override:
            return Path(str(override)).expanduser().resolve()
        return self._default_binary()

    # ---------------------------------------------------------------- #
    # Core solver call                                                   #
    # ---------------------------------------------------------------- #

    def _run_solver(
        self,
        instance_text: str,
        time_limit_sec: int,
    ) -> Tuple[int, bool, str]:
        """
        Write *instance_text* to a temp file, run ``main-solve``, parse output.

        Returns
        -------
        (n_relocations, optimal_proven, stdout_tail)
        """
        binary = self._resolve_binary()
        if not binary.is_file():
            raise FileNotFoundError(
                f"Jin solver binary not found at {binary}. "
                "Run `make` in vendor/ or set `jin_binary` / env "
                "`JIN_UCRP_BINARY`."
            )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jin_crpu", delete=False, encoding="utf-8"
        ) as tf:
            tf.write(instance_text)
            tmp_path = tf.name

        try:
            cmd = [
                str(binary),
                "--input",  tmp_path,
                "--time_limit", str(int(time_limit_sec)),
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

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        blob   = stdout + "\n" + stderr

        return self._parse_output(stdout, blob)

    def _parse_output(
        self, stdout: str, blob: str
    ) -> Tuple[int, bool, str]:
        matches = self._STAT_RE.findall(stdout)
        if matches:
            best_lb = int(matches[-1][0])
            best_ub = int(matches[-1][1])
            optimal = best_lb == best_ub
            return best_ub, optimal, blob[-4000:]

        # Fallback: count moves in the solution line.
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line == "?":
                raise RuntimeError(
                    "Jin solver returned '?' — no feasible solution found.\n"
                    f"---\n{blob[-4000:]}"
                )
            if line.startswith("[") and "->" in line:
                n_moves = line.count("->")
                return n_moves, False, blob[-4000:]

        raise RuntimeError(
            "Jin solver produced no parseable output.\n"
            f"returncode={blob[:200]}\n---\n{blob[-4000:]}"
        )

    # ---------------------------------------------------------------- #
    # BaseAlgorithm.train                                                #
    # ---------------------------------------------------------------- #

    def train(
        self,
        problem_factory: Callable,
        result_queue: mp.Queue,
        stop_event: mp.Event,
    ) -> None:
        cfg_a   = self.config
        n_seeds = max(1, cfg_a.num_eval_seeds)
        extra   = cfg_a.extra or {}
        t_limit = int(extra.get("jin_time_limit_sec", 600))

        all_metrics: List[Dict[str, float]] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            trace_layout(f"JinTanaka2023ForCRPU.train: seed {seed + 1}/{n_seeds}")

            env = problem_factory()
            env.config.seed = seed
            env.reset(seed=seed, options={"skip_auto_retrieve": True})

            instance_text = yard_to_jin_instance(env.config, env.yard)
            n_reloc, optimal, _dbg = self._run_solver(instance_text, t_limit)

            metrics = {
                "relocations":       float(n_reloc),
                "steps":             float(n_reloc),
                "time":              float(n_reloc),
                "optimal_proven":    1.0 if optimal else 0.0,
                "jin_time_limit_sec": float(t_limit),
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
                metric   = agg.get("relocations", 0.0),
                metrics  = agg,
                progress = 1.0,
                snapshot = {},
            )
