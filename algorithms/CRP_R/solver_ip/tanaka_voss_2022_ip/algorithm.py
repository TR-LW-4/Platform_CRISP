"""
TanakaVossIP2022
<2022> <exact> <restricted> <single-bay> <CRP-R>
IP-based exact algorithm (Algorithm 1) wrapper
tanaka2022ip_time_limit_sec --- 3600 --- Wall-clock limit per instance (s)
tanaka2022ip_n_threads --- 1 --- Gurobi threads (0 = auto)

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
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.layout_trace import trace_layout
from core.plan import Movement, RelocationPlan
from core.yard import Yard
from algorithms.CRP_R.solver_ip.common import (
    merge_solver_and_plan_metrics,
    plan_to_moves_extra,
    priority_to_container_id,
    sequential_retrieval_plan,
    stack_keys,
)

from .tanaka_export import yard_to_tanaka_instance

from .model import (
    _plan_from_tanaka_relocations,
)


class TanakaVossIP2022(BaseAlgorithm):

    name                = "Tanaka & Voß (2022) IP"
    category            = "Exact"
    requires_solver     = True
    solver_backend      = "gurobi"
    description         = "Tanaka & Voß (EJOR 2022) IP-based exact algorithm."
    compatible_problems = ["CRP-R"]
    geometry            = "single-bay"
    objectives          = ["relocations"]
    fidelity            = "faithful"
    # Output patterns from solve.cpp (written to stderr)
    _OPT_RE   = re.compile(r"optimal_value=(\d+)")
    _UB_RE    = re.compile(r"upper_bound=(\d+)")
    _RELOC_RE = re.compile(
        r"Relocation\s+\d+:\s*\[\s*(\d+)\s*\]\s*(\d+)->(\d+)"
    )
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
    ) -> Tuple[Optional[int], bool, str, List[Tuple[int, int, int]]]:
        """
        Run ``rbrp_ip`` on *instance_text* and parse results.

        Returns
        -------
        (relocations, optimal_proven, debug_blob, reloc_records)
            relocations   : int if a feasible solution was found, else None
            optimal_proven: True iff lb == ub (proven optimal)
            debug_blob    : tail of combined stderr+stdout for diagnostics
            reloc_records : list of (priority, src_stack, dst_stack) 1-indexed
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
        reloc_records = [
            (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            for m in self._RELOC_RE.finditer(blob)
        ]
        debug = blob[-6000:]

        m_opt = self._OPT_RE.search(blob)
        if m_opt:
            return int(m_opt.group(1)), True, debug, reloc_records

        m_ub = self._UB_RE.search(blob)
        if m_ub:
            return int(m_ub.group(1)), False, debug, reloc_records

        if "upper_bound=infeasible" in blob:
            return None, False, debug, reloc_records

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
                n_reloc, optimal, debug, reloc_records = self._run_ip(
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

            plan = None
            if n_reloc == 0:
                plan = sequential_retrieval_plan(env.yard)
            elif n_reloc is not None and reloc_records:
                plan = _plan_from_tanaka_relocations(env.yard, reloc_records)

            solver = {
                "obj": n_reloc,
                "optimal": optimal,
                "time_out": False,
                "n_vars": 0,
                "n_constrs": 0,
                "solve_time": 0.0,
            }
            if plan is not None:
                metrics = merge_solver_and_plan_metrics(
                    solver, env.validate_plan(plan),
                )
            else:
                metrics = merge_solver_and_plan_metrics(
                    solver,
                    {
                        "relocations": float(n_reloc) if n_reloc is not None else float("inf"),
                        "crane_time": float("inf"),
                        "time": float("inf") if n_reloc is None else 0.0,
                        "feasible": 0.0 if n_reloc is None else 1.0,
                        "completed": 0.0 if n_reloc is None else 1.0,
                        "validated": 0.0,
                    },
                )
            metrics["tanaka2022ip_time_limit_sec"] = float(t_limit)
            all_metrics.append(metrics)
            primary = float(metrics.get("relocations", float("inf")))

            if primary < self._best_metric:
                self._best_metric = primary
                self._best_solution = [
                    (m.to_pos[0] - 1) * env.config.num_rows + (m.to_pos[1] - 1)
                    for m in (plan.movements if plan is not None else [])
                    if m.to_pos is not None
                ]

            self._push(
                result_queue,
                step     = seed + 1,
                metric   = primary,
                metrics  = metrics,
                progress = (seed + 1) / n_seeds,
                extra    = {
                    "moves": plan_to_moves_extra(plan) if plan is not None else [],
                    "validation_errors": env.get_last_validation_errors(),
                    "debug_tail": debug,
                },
            )
            print(
                f"[TanakaVossIP2022] seed={seed}  "
                f"reloc={metrics.get('relocations')}  "
                f"crane={metrics.get('crane_time')}  "
                f"optimal={optimal}  "
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
