"""
Tanaka & Takii (2016) — Restricted-duplicate branch-and-bound for CRP-D.

Paper
-----
Shunji Tanaka and Kenta Takii,
"A Faster Branch-and-Bound Algorithm for the Block Relocation Problem,"
IEEE Transactions on Automation Science and Engineering,
vol. 13, no. 1, pp. 181–190, January 2016.
DOI: 10.1109/TASE.2015.2434417

Problem variant
---------------
Restricted BRP with **duplicate** priorities (CRP-Dr / "duplicate" in paper).
- Retrieval order: by group; within a group any accessible block may leave.
- Relocation rule: only the top of the stack above the currently selected
  target-group member may be relocated (restricted).
- Objective: minimise total relocations.

Algorithm
---------
Iterative branch-and-bound (§VI):
  1. Greedy upper-bound heuristic (§VI-C).
  2. B&B with depth-first search; target block chosen by branching (§VI-A).
  3. Lower bound: LB4e (§V-B) — extension of LB4 to duplicate priorities.
     LB4 is the paper's primary contribution over prior LB3 (Zhu et al. 2012).

Vendor binary
-------------
``vendor/restricted-duplicate-1.02/brp_bb``  (BSD licence, Shunji Tanaka).
Compile with ``make`` inside that directory before first use.
Override via ``extra["tanaka_binary"]`` or env var ``TANAKA_CRPD_BB``.

CLI interface (from main.c)
---------------------------
  brp_bb [-v|-s] [-S S] [-T T] [-t L] <input_file>
  -s : silent mode (suppress verbose output; stderr still has opt=/best=)
  -T T: max tiers (stack height cap)
  -t L: time limit in seconds

stdout: full solution trace + ``relocations=N``
stderr: ``opt=N`` (proven optimal) or ``best=N`` (time-limited)

Coupling
--------
No imports from any other algorithm in the platform.  Depends only on
``core.base_algorithm`` and this package's ``tanaka_export``.
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


class Tanaka2016BBDuplicate(BaseAlgorithm):
    """
    Shunji Tanaka & Kenta Takii (2016) — exact B&B for CRP-D (duplicate).

    Vendor binary: ``vendor/restricted-duplicate-1.02/brp_bb``.
    Solves the **restricted** BRP with duplicate priorities using LB4e.

    Configuration (via config.extra)
    ---------------------------------
    tanaka_time_limit_sec : int   CPU-time limit passed to brp_bb (default 600).
    tanaka_binary         : str   Path to brp_bb; empty = use bundled binary.
    num_eval_seeds        : int   Number of independent instances to evaluate.
    """

    name                = "Tanaka (2016) B&B [CRP-D]"
    category            = "Exact"
    description         = (
        "Exact branch-and-bound for restricted BRP with duplicate priorities "
        "(Tanaka & Takii, IEEE TASE 2016).  "
        "Uses LB4e — the tightest published lower bound for this variant. "
        "Wraps vendor binary ``restricted-duplicate-1.02/brp_bb``. "
        "Primary metric = relocations; ``optimal_proven=1`` when opt= is reported."
    )
    compatible_problems = ["CRP-D"]
    step_label          = "Seed"

    _OPT_RE  = re.compile(r"opt=(\d+)")
    _BEST_RE = re.compile(r"best=(\d+)")

    def __init__(self, config: Optional[AlgorithmConfig] = None) -> None:
        super().__init__(config)

    # ------------------------------------------------------------------ #
    # Binary resolution                                                    #
    # ------------------------------------------------------------------ #

    def _default_binary(self) -> Path:
        return (
            Path(__file__).resolve().parent
            / "vendor"
            / "restricted-duplicate-1.02"
            / "brp_bb"
        )

    def _resolve_binary(self) -> Path:
        extra    = self.config.extra or {}
        override = extra.get("tanaka_binary") or os.environ.get("TANAKA_CRPD_BB")
        if override:
            return Path(str(override)).expanduser().resolve()
        return self._default_binary()

    # ------------------------------------------------------------------ #
    # Run binary                                                           #
    # ------------------------------------------------------------------ #

    def _run_bb(
        self,
        instance_text:  str,
        max_tiers:      int,
        time_limit_sec: int,
    ) -> Tuple[int, bool, str]:
        """
        Call brp_bb on *instance_text*.

        Returns
        -------
        (n_relocations, optimal_proven, debug_tail)
        """
        binary = self._resolve_binary()
        if not binary.is_file():
            raise FileNotFoundError(
                f"Tanaka CRP-D solver binary not found at {binary}. "
                "Run `make` in vendor/restricted-duplicate-1.02, or set "
                "`tanaka_binary` in config.extra / env TANAKA_CRPD_BB."
            )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tanaka", delete=False, encoding="utf-8",
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

        blob   = (proc.stderr or "") + "\n" + (proc.stdout or "")
        m_opt  = self._OPT_RE.search(blob)
        m_best = self._BEST_RE.search(blob)
        if m_opt:
            return int(m_opt.group(1)), True, blob[-4000:]
        if m_best:
            return int(m_best.group(1)), False, blob[-4000:]
        raise RuntimeError(
            "Tanaka brp_bb (CRP-D) produced no opt=/best= line.\n"
            f"returncode={proc.returncode}\n---\n{blob[-8000:]}"
        )

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        cfg_a   = self.config
        extra   = cfg_a.extra or {}
        n_seeds = max(1, cfg_a.num_eval_seeds)
        t_limit = int(extra.get("tanaka_time_limit_sec", 600))

        all_metrics: List[Dict[str, float]] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            trace_layout(
                f"Tanaka2016BBDuplicate.train: seed {seed + 1}/{n_seeds}"
            )

            env = problem_factory()
            env.config.seed = seed
            env.reset(seed=seed, options={"skip_auto_retrieve": True})

            text = yard_to_tanaka_instance(env.config, env.yard)
            n_reloc, optimal, _dbg = self._run_bb(
                text,
                max_tiers      = env.config.max_tiers,
                time_limit_sec = t_limit,
            )

            metrics: Dict[str, float] = {
                "relocations":            float(n_reloc),
                "steps":                  float(n_reloc),
                "time":                   0.0,
                "optimal_proven":         1.0 if optimal else 0.0,
                "tanaka_time_limit_sec":  float(t_limit),
            }
            all_metrics.append(metrics)
            primary = float(n_reloc)

            if primary < self._best_metric:
                self._best_metric   = primary
                self._best_solution = []   # exact solver does not replay env

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

    # ------------------------------------------------------------------ #
    # Best solution                                                        #
    # ------------------------------------------------------------------ #

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    # ------------------------------------------------------------------ #
    # GUI configuration schema                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def config_schema(cls) -> Dict[str, Any]:
        base = super().config_schema()
        base.update({
            "num_eval_seeds": {
                "type":    "int",
                "default": 1,
                "min":     1,
                "max":     100,
                "label":   "Evaluation seeds",
                "help":    "Number of independent instances to evaluate.",
            },
            "tanaka_time_limit_sec": {
                "type":    "int",
                "default": 600,
                "min":     1,
                "max":     86400,
                "label":   "B&B time limit (s)",
                "help":    "CPU-time limit passed to brp_bb via -t.",
            },
            "tanaka_binary": {
                "type":    "str",
                "default": "",
                "label":   "Path to brp_bb (optional)",
                "help": (
                    "Leave empty to use bundled vendor/restricted-duplicate-1.02/brp_bb. "
                    "Override or set env TANAKA_CRPD_BB to use a custom build."
                ),
            },
        })
        return base
