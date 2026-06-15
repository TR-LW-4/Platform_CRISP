"""
Bacci, Mattia, Ventura (2020)
"A branch-and-cut algorithm for the restricted Block Relocation Problem"
European Journal of Operational Research 287 (2020) 452–459.

Algorithm: BC-RBRP
-------------------
Branch-and-cut exact algorithm based on a new time-period-indexed ILP
formulation for the restricted BRP.

Formulation overview
--------------------
Variables (1-indexed, period t = retrieval of block t):
  x[i,j,t]  = 1 if block i is in stack j at the start of period t
  y[i,j,t]  = 1 if block i is reshuffled from stack j during period t

Key constraints:
  (2)  Each block is in exactly one stack per period.
  (3)  Stack capacity limit.
  (4–7) Define y from x transitions.
  (8–9) Exponential ordering constraints (generated as cuts by callback).
  (13–18) Pre-processing / valid inequalities from initial layout.

Initial warm start: BBS heuristic (Bacci et al., 2019) run for 1 second.
Cuts: violated (8)/(9) constraints are separated when integer solutions are
found (see rBRP_MIP3.h callback).

Performance vs. Caserta benchmark Set 1:
  All instances with n ≤ 50 solved within seconds.
  n = 60–100: solved within seconds to 1 min on average.
  n = 100 (10×12): not solved within 1 hour.

Vendor C++ source (vendor/BC_RBRP/)
-------------------------------------
  BC_RBRP.cpp    : main entry point
  rBRP_MIP3.cpp  : IP model + separation callback (Gurobi C++ API)
  rBRP_MIP3.h    : callback class (cuts integer solutions violating (8)/(9))
  rBRP_BSheu.cpp : BBS warm start heuristic

Compilation
-----------
1. Edit vendor/BC_RBRP/Makefile:
     LIBGUROBI = /path/to/gurobi/linux64/lib -lgurobi_c++ -lgurobi<ver>
     INCGUROBI = /path/to/gurobi/linux64/include
2. cd vendor/BC_RBRP && make
3. Produces BC_RBRP.exe  (Linux: executable without extension needed)

Binary interface
----------------
Usage: ./BC_RBRP.exe instanceFile <solS=N> <optS=N> <verbS=N>
  solS=0   BC-RBRP exact (default)
  solS=1   BBS heuristic only
  optS=1   solve integer problem to optimality (default)
  optS=0   solve LP relaxation only
  verbS=1  print solution detail

stdout:
  "Heuristic solution value = N"        (BBS warm-start value)
  "Optimal solution = M, found in T secs."  (BC-RBRP result)

Note: the time limit is hardcoded in rBRP_MIP3.cpp to 3600 seconds.
To change it, edit and recompile.

Input format (BC_RBRP / BBS format)
-------------------------------------
Line 1: w h n        (stacks, max_tiers, containers)
Remaining w lines, one per stack left→right, bay-major order:
  k p1 p2 ... pk     (height k; priorities bottom→top, 1=first retrieved)
Empty stacks: "0"
"""

from __future__ import annotations

import multiprocessing as mp
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.layout_trace import trace_layout

from .bbs_export import yard_to_bbs_instance


class BacciBC2020(BaseAlgorithm):
    """
    Bacci, Mattia & Ventura (2020) — BC-RBRP branch-and-cut exact solver.

    Wraps ``vendor/BC_RBRP/BC_RBRP.exe`` (``solS=0``).  The binary uses
    Gurobi internally; compile with Gurobi headers/libraries before use.

    Compilation:
        cd vendor/BC_RBRP
        # edit Makefile: set LIBGUROBI and INCGUROBI to your Gurobi paths
        make
        # produces BC_RBRP.exe

    Note: the time limit is hardcoded to 3600 s in ``rBRP_MIP3.cpp``.

    Config parameters (``AlgorithmConfig.extra``)
    ---------------------------------------------
    bacci2020_binary  : str   override path to compiled binary (optional)
    """

    name                = "Bacci (2020) BC-RBRP [Gurobi]"
    category            = "Exact"
    requires_solver     = True
    solver_backend      = "gurobi"
    description         = (
        "Branch-and-cut exact algorithm for restricted BRP. "
        "Bacci, Mattia & Ventura — EJOR 287 (2020). "
        "New time-period-indexed ILP + exponential cuts via Gurobi callback. "
        "Outperforms CRP-I and Tanaka B&B on Caserta benchmarks. "
        "[Requires Gurobi: compile vendor/BC_RBRP/BC_RBRP.exe first]"
    )
    compatible_problems = ["CRP-R"]
    step_label          = "Seed"

    # Parse BBS warm-start value and BC-RBRP result from stdout
    _BBS_RE = re.compile(r"Heuristic solution value\s*=\s*(\d+)")
    _OPT_RE = re.compile(r"Optimal solution\s*=\s*([\d.]+),\s*found in\s*([\d.]+)\s*secs")

    def __init__(self, config: Optional[AlgorithmConfig] = None) -> None:
        super().__init__(config)

    # ---------------------------------------------------------------- #
    # Binary resolution                                                  #
    # ---------------------------------------------------------------- #

    def _default_binary(self) -> Path:
        return (
            Path(__file__).resolve().parent
            / "vendor" / "BC_RBRP" / "BC_RBRP.exe"
        )

    def _resolve_binary(self) -> Path:
        extra    = self.config.extra or {}
        override = (
            extra.get("bacci2020_binary")
            or os.environ.get("BACCI2020_BC_RBRP")
        )
        if override:
            return Path(str(override)).expanduser().resolve()
        return self._default_binary()

    # ---------------------------------------------------------------- #
    # Core subprocess call                                               #
    # ---------------------------------------------------------------- #

    def _run_bc(
        self,
        instance_text: str,
        timeout_s: float = 4200.0,   # 3600 s (hardcoded limit) + buffer
    ) -> Tuple[Optional[int], Optional[int], float]:
        """
        Write instance to temp file, run BC_RBRP.exe (solS=0, optS=1),
        parse and return (bc_reloc, bbs_reloc, solve_time_s).

        bc_reloc  : int   BC-RBRP result (None if binary not found or failed)
        bbs_reloc : int   BBS warm-start value (None if not printed)
        solve_time: float seconds reported by the binary
        """
        binary = self._resolve_binary()
        if not binary.is_file():
            raise FileNotFoundError(
                f"BC_RBRP binary not found at {binary}.\n"
                "Compile the vendor source:\n"
                "  cd vendor/BC_RBRP\n"
                "  # edit Makefile: set LIBGUROBI / INCGUROBI\n"
                "  make\n"
                "Or set env var BACCI2020_BC_RBRP to the binary path."
            )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as tf:
            tf.write(instance_text)
            tmp_path = tf.name

        try:
            proc = subprocess.run(
                [str(binary), tmp_path, "solS=0", "optS=1", "verbS=0"],
                capture_output=True,
                text=True,
                timeout=float(timeout_s),
                check=False,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        stdout = (proc.stdout or "") + "\n" + (proc.stderr or "")

        bbs_val: Optional[int]  = None
        bc_val:  Optional[int]  = None
        solve_t: float          = 0.0

        m_bbs = self._BBS_RE.search(stdout)
        if m_bbs:
            bbs_val = int(m_bbs.group(1))

        m_opt = self._OPT_RE.search(stdout)
        if m_opt:
            bc_val  = int(round(float(m_opt.group(1))))
            solve_t = float(m_opt.group(2))

        return bc_val, bbs_val, solve_t

    # ---------------------------------------------------------------- #
    # BaseAlgorithm interface                                            #
    # ---------------------------------------------------------------- #

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        cfg_a   = self.config
        n_seeds = max(1, cfg_a.num_eval_seeds)
        extra   = cfg_a.extra or {}

        all_metrics: List[Dict] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            trace_layout(f"BacciBC2020.train: seed {seed + 1}/{n_seeds}")

            env = problem_factory()
            env.config.seed = seed
            env.reset(seed=seed, options={"skip_auto_retrieve": True})

            instance_text = yard_to_bbs_instance(env.config, env.yard)

            try:
                bc_reloc, bbs_reloc, solve_t = self._run_bc(instance_text)
            except FileNotFoundError as exc:
                print(f"[BacciBC2020] {exc}", file=sys.stderr, flush=True)
                self._push(
                    result_queue,
                    step     = seed + 1,
                    metric   = float("inf"),
                    metrics  = {"relocations": float("inf"), "binary_missing": 1.0},
                    progress = (seed + 1) / n_seeds,
                )
                continue

            primary = float(bc_reloc) if bc_reloc is not None else float("inf")

            if primary < self._best_metric:
                self._best_metric   = primary
                self._best_solution = []

            metrics: Dict = {
                "relocations":    primary,
                "steps":          primary,
                "time":           primary,
                "bbs_warm_start": float(bbs_reloc) if bbs_reloc is not None else float("inf"),
                "solve_time_s":   round(solve_t, 3),
                "feasible":       0.0 if bc_reloc is None else 1.0,
                # BC-RBRP's own output says "Optimal solution" but we cannot
                # distinguish proven-optimal from time-limit here (status not printed).
                # Mark as likely optimal (the paper solves n≤50 trivially).
                "optimal_likely": 1.0 if bc_reloc is not None and solve_t < 3590 else 0.0,
            }
            all_metrics.append(metrics)

            self._push(
                result_queue,
                step     = seed + 1,
                metric   = primary,
                metrics  = metrics,
                progress = (seed + 1) / n_seeds,
                snapshot = env.get_state_snapshot(),
            )
            print(
                f"[BacciBC2020] seed={seed}  N={len(env.containers)}  "
                f"bc={bc_reloc}  bbs={bbs_reloc}  t={solve_t:.1f}s",
                file=sys.stderr, flush=True,
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
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "num_eval_seeds": {
                "type": "int", "default": 1, "min": 1, "max": 100,
                "label": "Seeds",
            },
            "bacci2020_binary": {
                "type": "str", "default": "",
                "label": "Path to BC_RBRP.exe (optional override)",
                "help": (
                    "Leave empty to use vendor/BC_RBRP/BC_RBRP.exe. "
                    "Can also be set via env var BACCI2020_BC_RBRP. "
                    "Note: time limit is hardcoded to 3600 s in the source; "
                    "edit rBRP_MIP3.cpp and recompile to change it."
                ),
            },
        })
        return base
