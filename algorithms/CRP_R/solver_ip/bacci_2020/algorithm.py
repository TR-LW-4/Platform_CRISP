"""
BacciBC2020
<2020> <exact> <restricted> <single-bay> <CRP-R>
BC-RBRP branch-and-cut IP wrapper
bacci2020_binary ---  --- Path to BC_RBRP.exe (optional)

------------------------------- Reference --------------------------------
T. Bacci, S. Mattia, P. Ventura,
"A branch-and-cut algorithm for the restricted Block Relocation Problem",
European Journal of Operational Research 287 (2020) 452–459.
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
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.layout_trace import trace_layout
from algorithms.CRP_R.solver_ip.common import (
    merge_solver_and_plan_metrics,
    occupancy_from_yard,
    occupancy_stages_to_plan,
    plan_to_moves_extra,
    priority_to_container_id,
    sequential_retrieval_plan,
    stack_keys,
)

from .bbs_export import yard_to_bbs_instance
from .model import parse_bacci_period_layouts


class BacciBC2020(BaseAlgorithm):

    name                = "Bacci (2020) BC-RBRP [Gurobi]"
    category            = "Exact"
    requires_solver     = True
    solver_backend      = "gurobi"
    description         = "Bacci et al. (EJOR 2020) BC-RBRP branch-and-cut IP."
    compatible_problems = ["CRP-R"]
    geometry            = "single-bay"
    objectives          = ["relocations"]
    fidelity            = "faithful"
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
    ) -> Tuple[Optional[int], Optional[int], float, str]:
        """
        Write instance to temp file, run BC_RBRP.exe (solS=0, optS=1, verbS=1),
        parse and return (bc_reloc, bbs_reloc, solve_time_s, stdout).
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
                [str(binary), tmp_path, "solS=0", "optS=1", "verbS=1"],
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

        return bc_val, bbs_val, solve_t, stdout

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
        n_seeds = 1  # multi-seed eval removed; single run only
        extra   = cfg_a.extra or {}

        all_metrics: List[Dict] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            trace_layout(f"BacciBC2020.train: seed {seed + 1}/{n_seeds}")

            env = problem_factory()
            env.reset(seed=seed, options={"skip_auto_retrieve": True})

            instance_text = yard_to_bbs_instance(env.config, env.yard)

            try:
                bc_reloc, bbs_reloc, solve_t, stdout = self._run_bc(instance_text)
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

            n_cont = len(env.containers)
            w = env.config.num_bays * env.config.num_rows
            h = env.config.max_tiers
            keys = stack_keys(env.yard)
            pri = priority_to_container_id(env.yard)
            layouts = parse_bacci_period_layouts(stdout, w, h)
            plan = None
            if bc_reloc == 0:
                plan = sequential_retrieval_plan(env.yard)
            elif bc_reloc is not None:
                occupancy = layouts if layouts else {1: occupancy_from_yard(env.yard)}
                plan = occupancy_stages_to_plan(
                    occupancy, {}, n_cont, keys, pri, max_tiers=h,
                )

            solver = {
                "obj": bc_reloc,
                "optimal": bool(bc_reloc is not None and solve_t < 3590),
                "time_out": bool(bc_reloc is not None and solve_t >= 3590),
                "n_vars": 0,
                "n_constrs": 0,
                "solve_time": solve_t,
            }
            if plan is not None:
                metrics = merge_solver_and_plan_metrics(
                    solver, env.validate_plan(plan),
                )
            else:
                metrics = merge_solver_and_plan_metrics(
                    solver,
                    {
                        "relocations": float(bc_reloc) if bc_reloc is not None else float("inf"),
                        "crane_time": float("inf"),
                        "time": float("inf") if bc_reloc is None else 0.0,
                        "feasible": 0.0 if bc_reloc is None else 1.0,
                        "completed": 0.0 if bc_reloc is None else 1.0,
                        "validated": 0.0,
                    },
                )
            metrics["bbs_warm_start"] = (
                float(bbs_reloc) if bbs_reloc is not None else float("inf")
            )
            metrics["optimal_likely"] = (
                1.0 if bc_reloc is not None and solve_t < 3590 else 0.0
            )
            metrics["solve_time_s"] = round(solve_t, 3)
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
                },
            )
            print(
                f"[BacciBC2020] seed={seed}  N={len(env.containers)}  "
                f"reloc={metrics.get('relocations')}  "
                f"crane={metrics.get('crane_time')}  "
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
