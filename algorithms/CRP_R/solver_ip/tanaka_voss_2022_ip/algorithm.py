"""
Tanaka & Voß (2022) — IP-based exact algorithm for restricted BRP.

Paper
-----
S. Tanaka and S. Voß,
"An exact approach to the restricted block relocation problem based on
a new integer programming formulation,"
European Journal of Operational Research, 296(2):485-503, 2022.

This module wraps the **primary Algorithm 1** from the paper
(as opposed to ``tanaka_voss_2022`` B&B, which wraps the comparison B&B).

Algorithm overview (Sections 3–4)
----------------------------------
1. Enumerate relocation sequences for each blocking block.
2. Build a relaxed IP (RP) using truncated sequences → lower bound.
3. Build an upper-bound IP (UP) using complete sequences.
4. Iteratively expand truncated sequences and re-solve until gap = 0.

The binary ``rbrp_ip`` uses **Gurobi** internally to solve each RP/UP
subproblem.  It is the first algorithm to solve all Caserta benchmark
instances with ≤ 100 blocks to proven optimality.

Vendor C++ source
-----------------
vendor/restricted-distinct-ip-1.0/
    bay.cpp / baystate.cpp     : bay state representation
    sequence.cpp / ipmodel.cpp : sequence enumeration + Gurobi IP model
    greedy.cpp                 : greedy upper-bound initialisation
    solve.cpp / main.cpp       : main loop (Algorithm 1)
    Makefile                   : compile with ``make``

Compilation
-----------
Requires Gurobi C++ headers/libraries and Boost (program_options).

  1. Edit ``Makefile``:
       GUROBI_ROOT = /path/to/your/gurobi/linux64
       GUROBI_LIBS = -lgurobi_g++5.2 -lgurobi<version>
  2. ``cd vendor/restricted-distinct-ip-1.0 && make``
  3. This produces ``rbrp_ip``.

Binary interface
----------------
Usage: rbrp_ip [options] <input_file>
   -T N   maximum number of tiers (height limit)
   -E N   number of empty tiers (alternative to -T)
   -t N   time limit in seconds (float)
   -m N   number of threads (Gurobi parameter)
   -s N   threshold for sequence expansion (default 100)
   -v N   verbose level
   -g     disable greedy upper bound
   -u     disable upper bounding

stdout : solution move list (if feasible)
stderr : diagnostic lines including:
           solved  /  not solved
           optimal_value=N          (when lb == ub → proven optimal)
           lower_bound=N            (best lb when not proven)
           upper_bound=N            (best ub / best solution)
           upper_bound=infeasible   (if no feasible solution found)
           iterations=N
           total_time=N.NNN
           ...

Input format
------------
Identical to other Tanaka solvers (Caserta format):
    Line 1:  <n_stacks> <n_blocks>
    Line 2+: for each stack: <height> <p_1> ... <p_h>  (bottom → top)
"""

from __future__ import annotations

import multiprocessing as mp
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.layout_trace import trace_layout

from .tanaka_export import yard_to_tanaka_instance


class TanakaVossIP2022(BaseAlgorithm):
    """
    Tanaka & Voß (2022) IP-based exact solver — Algorithm 1.

    Wraps the ``rbrp_ip`` binary (vendor/restricted-distinct-ip-1.0).

    ``rbrp_ip`` must be compiled before use:
        cd vendor/restricted-distinct-ip-1.0
        # edit Makefile: set GUROBI_ROOT and GUROBI_LIBS
        make

    Alternatively, set ``extra["tanaka2022ip_binary"]`` or the environment
    variable ``TANAKA2022IP_RBRP_IP`` to the full path of a pre-built binary.

    Key parameters (``AlgorithmConfig.extra``)
    ------------------------------------------
    tanaka2022ip_time_limit_sec : int   wall-clock limit per instance (default 3600)
    tanaka2022ip_n_threads      : int   Gurobi threads; 0 = auto (default 1)
    tanaka2022ip_threshold      : int   sequence-expansion threshold (default 100)
    tanaka2022ip_binary         : str   override binary path (optional)
    """

    name                = "Tanaka & Voß (2022) IP"
    category            = "Exact"
    requires_solver     = True
    solver_backend      = "gurobi"
    description         = (
        "IP-based exact algorithm (Algorithm 1) for restricted BRP with distinct priorities. "
        "Tanaka & Voß — EJOR 296 (2022). "
        "First method to solve all ≤100-block Caserta instances to optimality. "
        "Requires Gurobi (via compiled rbrp_ip binary). "
        "[Requires Gurobi license]"
    )
    compatible_problems = ["CRP-R"]
    # Output patterns from solve.cpp (written to stderr)
    _OPT_RE   = re.compile(r"optimal_value=(\d+)")
    _UB_RE    = re.compile(r"upper_bound=(\d+)")
    _LB_RE    = re.compile(r"lower_bound=(\d+)")
    _SOLVED   = re.compile(r"^solved$", re.MULTILINE)
    _TIME_RE  = re.compile(r"total_time=([\d.]+)")
    _ITER_RE  = re.compile(r"iterations=(\d+)")

    def __init__(self, config: Optional[AlgorithmConfig] = None) -> None:
        super().__init__(config)

    # ---------------------------------------------------------------- #
    # Binary resolution                                                   #
    # ---------------------------------------------------------------- #

    def _default_binary(self) -> Path:
        return (
            Path(__file__).resolve().parent
            / "vendor"
            / "restricted-distinct-ip-1.0"
            / "rbrp_ip"
        )

    def _resolve_binary(self) -> Path:
        extra    = self.config.extra or {}
        override = (
            extra.get("tanaka2022ip_binary")
            or os.environ.get("TANAKA2022IP_RBRP_IP")
        )
        if override:
            return Path(str(override)).expanduser().resolve()
        return self._default_binary()

    # ---------------------------------------------------------------- #
    # Core IP runner                                                      #
    # ---------------------------------------------------------------- #

    def _run_ip(
        self,
        instance_text: str,
        max_tiers: int,
        time_limit_sec: float,
        n_threads: int,
        threshold: int,
    ) -> Tuple[Optional[int], bool, str]:
        """
        Run ``rbrp_ip`` on *instance_text* and parse results.

        Returns
        -------
        (relocations, optimal_proven, debug_blob)
            relocations   : int if a feasible solution was found, else None
            optimal_proven: True iff lb == ub (proven optimal)
            debug_blob    : tail of combined stderr+stdout for diagnostics
        """
        binary = self._resolve_binary()
        if not binary.is_file():
            raise FileNotFoundError(
                f"rbrp_ip binary not found at {binary}.\n"
                "Compile the vendor source:\n"
                "  cd vendor/restricted-distinct-ip-1.0\n"
                "  # edit Makefile: set GUROBI_ROOT and GUROBI_LIBS\n"
                "  make\n"
                "Or set `tanaka2022ip_binary` / env TANAKA2022IP_RBRP_IP."
            )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".dat", delete=False, encoding="utf-8"
        ) as tf:
            tf.write(instance_text)
            tmp_path = tf.name

        try:
            cmd = [
                str(binary),
                "-T", str(int(max_tiers)),
                "-t", str(float(time_limit_sec)),
                "-m", str(int(n_threads)),
                "-s", str(int(threshold)),
                tmp_path,
            ]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=float(time_limit_sec) + 120.0,
                check=False,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        blob = (proc.stderr or "") + "\n" + (proc.stdout or "")

        # Parse optimal value (lb == ub)
        m_opt = self._OPT_RE.search(blob)
        if m_opt:
            return int(m_opt.group(1)), True, blob[-6000:]

        # Parse best upper bound (feasible but not proven optimal)
        m_ub = self._UB_RE.search(blob)
        if m_ub:
            return int(m_ub.group(1)), False, blob[-6000:]

        # No feasible solution found (e.g. time limit before first UB)
        if "upper_bound=infeasible" in blob:
            return None, False, blob[-6000:]

        raise RuntimeError(
            "rbrp_ip produced no optimal_value= / upper_bound= line.\n"
            f"returncode={proc.returncode}\n---\n{blob[-8000:]}"
        )

    # ---------------------------------------------------------------- #
    # BaseAlgorithm interface                                             #
    # ---------------------------------------------------------------- #

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        cfg_a   = self.config
        n_seeds = 1  # multi-seed eval removed; single run only
        extra   = cfg_a.extra or {}

        t_limit   = float(extra.get("tanaka2022ip_time_limit_sec", 3600))
        n_threads = int(extra.get("tanaka2022ip_n_threads", 1))
        threshold = int(extra.get("tanaka2022ip_threshold", 100))

        all_metrics: List[Dict[str, float]] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            trace_layout(
                f"TanakaVossIP2022.train: seed {seed + 1}/{n_seeds}  "
                f"threads={n_threads}  threshold={threshold}"
            )

            env = problem_factory()
            env.reset(seed=seed, options={"skip_auto_retrieve": True})

            text = yard_to_tanaka_instance(env.config, env.yard)

            try:
                n_reloc, optimal, debug = self._run_ip(
                    text,
                    max_tiers     = env.config.max_tiers,
                    time_limit_sec= t_limit,
                    n_threads     = n_threads,
                    threshold     = threshold,
                )
            except FileNotFoundError as exc:
                print(f"[TanakaVossIP2022] {exc}", file=sys.stderr, flush=True)
                self._push(
                    result_queue,
                    step     = seed + 1,
                    metric   = float("inf"),
                    metrics  = {"relocations": float("inf"), "binary_missing": 1.0},
                    progress = (seed + 1) / n_seeds,
                )
                continue

            primary = float(n_reloc) if n_reloc is not None else float("inf")

            if primary < self._best_metric:
                self._best_metric   = primary
                self._best_solution = []

            metrics: Dict[str, float] = {
                "relocations":    primary,
                "steps":          primary,
                "time":           0.0,
                "optimal_proven": 1.0 if optimal else 0.0,
                "feasible":       0.0 if n_reloc is None else 1.0,
                "tanaka2022ip_time_limit_sec": float(t_limit),
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
                f"[TanakaVossIP2022] seed={seed}  "
                f"reloc={n_reloc}  optimal={optimal}  "
                f"N={len(env.containers)}  S={env.config.num_bays * env.config.num_rows}",
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
    def config_schema(cls) -> Dict[str, Any]:
        base = super().config_schema()
        base.update({
            "tanaka2022ip_time_limit_sec": {
                "type": "float", "default": 3600.0, "min": 1.0, "max": 86400.0,
                "label": "Time limit (s)",
                "help": "Hard wall-clock limit per instance (passed to rbrp_ip -t).",
            },
            "tanaka2022ip_n_threads": {
                "type": "int", "default": 1, "min": 0, "max": 64,
                "label": "Gurobi threads (0 = auto)",
                "help": "Passed to rbrp_ip -m; controls Gurobi thread count.",
            },
            "tanaka2022ip_threshold": {
                "type": "int", "default": 100, "min": 1, "max": 10000,
                "label": "Sequence expansion threshold (-s)",
                "help": (
                    "Controls how many relocation sequences are expanded per iteration. "
                    "Higher values → fewer iterations but more memory/time per iteration."
                ),
            },
            "tanaka2022ip_binary": {
                "type": "str", "default": "",
                "label": "Path to rbrp_ip (optional override)",
                "help": (
                    "Leave empty to use vendor/restricted-distinct-ip-1.0/rbrp_ip. "
                    "Can also be set via env var TANAKA2022IP_RBRP_IP."
                ),
            },
        })
        return base
