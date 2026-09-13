"""
TanakaBB2019Stow
<2019> <exact> <stowage> <multi-bay> <CRP-Stow>
Iterative-deepening branch-and-bound for the BRPSP
variant --- unrestricted --- Restricted or unrestricted BRPSP
time_limit_sec --- 1800 --- Per-instance wall-clock limit (s)

------------------------------- Reference --------------------------------
S. Tanaka, S. Voß,
"An exact algorithm for the block relocation problem with a stowage plan",
European Journal of Operational Research 279 (2019) 767–781.
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
from core.benchmark_keys import layout_path_from_extra
from core.layout_trace import trace_layout

from .pro_export import env_to_pro_string


# ── regex patterns ─────────────────────────────────────────────────── #
_OPT_RE  = re.compile(r"opt=(\d+)")
_BEST_RE = re.compile(r"best=(\d+)")


class TanakaBB2019Stow(BaseAlgorithm):
    """
    Exact BRPSP solver from Tanaka & Voß (2019), wrapped as a platform
    algorithm for CRP-Stow.

    Key config options (set via ``extra`` dict or GUI):

    ``variant`` (str, default "unrestricted")
        Which BRPSP variant to solve: ``"unrestricted"`` or ``"restricted"``.
        Selects the pre-compiled binary brpsp_bb_u or brpsp_bb_r.

    ``time_limit_sec`` (int, default 1800)
        Hard wall-clock limit per instance, passed as ``-t`` to the solver.

    ``binary_u`` / ``binary_r`` (str, optional)
        Absolute path overrides for the two solver binaries.
        Can also be set via env vars ``TANAKA2019_BRPSP_BB_U`` /
        ``TANAKA2019_BRPSP_BB_R``.

    ``layout_file_path`` (str, optional, same key as CRP_Stow)
        When present, passes the .pro file directly to the solver without
        regenerating it from the env state.  This is the recommended mode
        for benchmark evaluation because it matches the file the problem
        was loaded from exactly.
    """

    name                = "Tanaka & Voß (2019) B&B"
    category            = "Exact"
    description         = (
        "Exact branch-and-bound with iterative deepening for the BRPSP "
        "(Tanaka & Voß 2019, EJOR 279:767-781). "
        "Three lower bounds: LB2c, LB2c4c, LBr (default). "
        "Supports unrestricted and restricted BRPSP variants. "
        "Input: Jovanović .pro benchmark files or random CRP-Stow episodes. "
        "Primary metric: relocations (optimal count when solver proves optimality)."
    )
    compatible_problems = ["CRP-Stow"]
    geometry            = "multi-bay"
    objectives          = ["relocations"]
    fidelity            = "faithful"
    step_label          = "Instance"

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    # ---------------------------------------------------------------- #
    # Binary resolution                                                  #
    # ---------------------------------------------------------------- #

    @property
    def _vendor_dir(self) -> Path:
        return Path(__file__).resolve().parent / "vendor" / "brpsp-1.0"

    def _resolve_binary(self, variant: str) -> Path:
        """Return path to the compiled binary for *variant* ("unrestricted"|"restricted")."""
        extra = self.config.extra or {}
        if variant == "restricted":
            override = (
                extra.get("binary_r")
                or os.environ.get("TANAKA2019_BRPSP_BB_R")
            )
            default = self._vendor_dir / "brpsp_bb_r"
        else:
            override = (
                extra.get("binary_u")
                or os.environ.get("TANAKA2019_BRPSP_BB_U")
            )
            default = self._vendor_dir / "brpsp_bb_u"

        return Path(str(override)).expanduser().resolve() if override else default

    # ---------------------------------------------------------------- #
    # Core solver call                                                   #
    # ---------------------------------------------------------------- #

    def _run_solver(
        self,
        input_path: str,
        variant: str,
        time_limit_sec: int,
        n_stacks: int,
        max_tiers: int,
    ) -> Tuple[int, bool, str]:
        """
        Run the brpsp_bb binary on *input_path* (.pro file).

        Returns
        -------
        (n_reloc, optimal_proven, debug_blob)
            n_reloc        : best relocation count found
            optimal_proven : True when solver output ``opt=N``
            debug_blob     : last ~4 KB of combined stderr+stdout
        """
        binary = self._resolve_binary(variant)
        if not binary.is_file():
            raise FileNotFoundError(
                f"brpsp_bb binary not found at {binary}.\n"
                f"Run  make  (or  make restrict  for the restricted variant) "
                f"in {binary.parent.parent.parent} "
                "or set config.extra['binary_u'/'binary_r'] / "
                "env var TANAKA2019_BRPSP_BB_U / TANAKA2019_BRPSP_BB_R."
            )

        cmd = [
            str(binary),
            "-s",                         # silent (no verbose yard printout)
            "-S", str(int(n_stacks)),
            "-T", str(int(max_tiers)),
            "-t", str(int(time_limit_sec)),
            str(input_path),
        ]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=float(time_limit_sec) + 120.0,
            check=False,
        )

        blob = (proc.stderr or "") + "\n" + (proc.stdout or "")
        m_opt  = _OPT_RE.search(blob)
        m_best = _BEST_RE.search(blob)

        if m_opt:
            return int(m_opt.group(1)), True, blob[-4000:]
        if m_best:
            return int(m_best.group(1)), False, blob[-4000:]

        raise RuntimeError(
            f"brpsp_bb produced no opt= / best= line.\n"
            f"returncode={proc.returncode}  binary={binary}\n"
            f"---\n{blob[-8000:]}"
        )

    def _get_pro_path(self, env) -> Tuple[str, bool]:
        """
        Return (path_to_pro_file, is_temp).

        If layout_file_path is in config.extra, return that path directly
        (no temp file).  Otherwise, write env state to a temp file and
        return its path (caller must delete it when is_temp=True).
        """
        path_str = layout_path_from_extra(env.config.extra)
        if path_str and Path(path_str).is_file():
            return path_str, False

        pro_text = env_to_pro_string(env)
        tf = tempfile.NamedTemporaryFile(
            mode="w", suffix=".pro", delete=False, encoding="utf-8"
        )
        tf.write(pro_text)
        tf.close()
        return tf.name, True

    # ---------------------------------------------------------------- #
    # Train (platform evaluation loop)                                   #
    # ---------------------------------------------------------------- #

    def train(
        self,
        problem_factory: Callable,
        result_queue: mp.Queue,
        stop_event: mp.Event,
    ) -> None:
        cfg_a   = self.config
        n_seeds = 1  # multi-seed eval removed; single run only
        extra   = cfg_a.extra or {}

        variant   = str(extra.get("variant", "unrestricted")).strip().lower()
        t_limit   = int(extra.get("time_limit_sec", 1800))

        all_metrics: List[Dict[str, float]] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            trace_layout(
                f"TanakaBB2019Stow.train: instance {seed + 1}/{n_seeds}  "
                f"variant={variant}  time_limit={t_limit}s"
            )

            env = problem_factory()
            env.reset(seed=seed)

            pro_path, is_temp = self._get_pro_path(env)
            try:
                n_reloc, optimal, _dbg = self._run_solver(
                    input_path    = pro_path,
                    variant       = variant,
                    time_limit_sec= t_limit,
                    n_stacks      = env._n_stacks,
                    max_tiers     = env.config.max_tiers,
                )
            finally:
                if is_temp:
                    try:
                        os.unlink(pro_path)
                    except OSError:
                        pass

            metrics = {
                "relocations":    float(n_reloc),
                "steps":          float(n_reloc),
                "time":           0.0,
                "optimal_proven": 1.0 if optimal else 0.0,
            }
            all_metrics.append(metrics)
            primary = float(n_reloc)

            if primary < self._best_metric:
                self._best_metric   = primary
                self._best_solution = []   # B&B produces a relocation plan, not gym actions

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

    # ---------------------------------------------------------------- #
    # Public interface                                                   #
    # ---------------------------------------------------------------- #

    def get_best_solution(self) -> Optional[List[int]]:
        """
        The B&B produces a relocation sequence (container moves), not a
        sequence of Gym action integers.  Returns an empty list as a sentinel.
        """
        return self._best_solution

    # ---------------------------------------------------------------- #
    # Config schema                                                      #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict[str, Any]:
        base = super().config_schema()
        base.update({
            "variant": {
                "type": "str", "default": "unrestricted",
                "label": "BRPSP variant",
                "help": (
                    "\"unrestricted\" (any stack-top may be relocated, brpsp_bb_u) "
                    "or \"restricted\" (only the topmost blocker, brpsp_bb_r). "
                    "Requires the corresponding binary to be compiled via make."
                ),
            },
            "time_limit_sec": {
                "type": "int", "default": 1800, "min": 1, "max": 86400,
                "label": "Time limit (s)",
                "help": "Passed to brpsp_bb -t.  Hard wall-clock limit per instance.",
            },
            "binary_u": {
                "type": "str", "default": "",
                "label": "Path to brpsp_bb_u (optional)",
                "help": (
                    "Absolute path override for the unrestricted variant binary. "
                    "Leave empty to use vendor/brpsp-1.0/brpsp_bb_u. "
                    "Can also be set via env var TANAKA2019_BRPSP_BB_U."
                ),
            },
            "binary_r": {
                "type": "str", "default": "",
                "label": "Path to brpsp_bb_r (optional)",
                "help": (
                    "Absolute path override for the restricted variant binary. "
                    "Leave empty to use vendor/brpsp-1.0/brpsp_bb_r. "
                    "Can also be set via env var TANAKA2019_BRPSP_BB_R."
                ),
            },
        })
        return base
