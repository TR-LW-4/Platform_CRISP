"""
KuArthanari2016Abstraction
<2016> <exact> <restricted> <single-bay> <CRP-R>
Abstraction method with pattern-database bidirectional search
time_limit_s --- 300 --- Per-instance search limit (s)
pdb_depth --- 5 --- PDB remaining-container depth

------------------------------- Reference --------------------------------
D. Ku, T.S. Arthanari,
"On the abstraction method for the container relocation problem",
Computers & Operations Research 68 (2016) 110–122.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import AlgorithmConfig, BaseAlgorithm
from core.yard import Yard

from .abstraction_core import build_pdb, solve


# ================================================================ #
#  Yard -> lightweight state conversion                              #
# ================================================================ #

def _yard_to_stacks(yard: Yard) -> List[List[int]]:
    """Return [[priority, ...], ...] bottom -> top, ordered by (bay, row)."""
    stack_list = sorted(yard.stacks.values(), key=lambda s: (s.bay, s.row))
    return [[c.priority for c in st.containers] for st in stack_list]


# ================================================================ #
#  BaseAlgorithm subclass                                            #
# ================================================================ #

class KuArthanari2016Abstraction(BaseAlgorithm):

    name = "Ku & Arthanari (2016) Abstraction+PDB"
    category = "Exact"
    description = "Ku & Arthanari (COR 2016) abstraction + pattern-database search."
    compatible_problems = ["CRP-R"]
    def __init__(self, config: Optional[AlgorithmConfig] = None) -> None:
        super().__init__(config)

    # ---------------------------------------------------------------- #
    # Train                                                              #
    # ---------------------------------------------------------------- #

    def train(
        self,
        problem_factory: Callable,
        result_queue: mp.Queue,
        stop_event: mp.Event,
    ) -> None:
        cfg = self.config
        extra = cfg.extra or {}
        time_limit = float(extra.get("time_limit_s", 300.0))
        pdb_depth = int(extra.get("pdb_depth", 5))
        cache_depth = int(extra.get("cache_depth", 20))
        max_cache_size = int(extra.get("max_cache_size", 1_000_000))
        pdb_build_time_limit = float(extra.get("pdb_build_time_limit_s", 60.0))

        n_seeds = 1  # multi-seed eval removed; single run only
        all_metrics: List[Dict] = []

        # ── Build the PDB once, amortised across every seed (Section 4:
        # "the time and space requirement for computing the PDB can be
        # amortised over solving many instances of the problem"). ──── #
        probe_env = problem_factory()
        probe_env.config.seed = 0
        probe_env.reset(seed=0, options={"skip_auto_retrieve": True})
        n_cols = probe_env.config.num_bays * probe_env.config.num_rows
        max_tiers_probe = probe_env.config.max_tiers

        t_pdb_start = time.perf_counter()
        pdb_levels = build_pdb(
            depth=pdb_depth,
            n_cols=n_cols,
            m_tiers=max_tiers_probe,
            time_limit_s=pdb_build_time_limit,
            stop_event=stop_event,
        )
        pdb_built_depth = max(pdb_levels.keys()) if pdb_levels else 0
        t_pdb_elapsed = time.perf_counter() - t_pdb_start

        print(
            f"[KuArthanari2016Abstraction] PDB built: depth={pdb_built_depth} "
            f"(requested {pdb_depth}), states={sum(len(v) for v in pdb_levels.values())}, "
            f"t={t_pdb_elapsed:.3f}s",
            file=sys.stderr,
            flush=True,
        )

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            t_start = time.perf_counter()

            env = problem_factory()
            env.reset(seed=seed, options={"skip_auto_retrieve": True})

            n_total = len(env.containers)
            max_tiers = env.config.max_tiers
            stacks = _yard_to_stacks(env.yard)

            result = solve(
                stacks, n_total, max_tiers,
                pdb_levels=pdb_levels,
                pdb_depth=pdb_built_depth,
                cache_depth=cache_depth,
                max_cache_size=max_cache_size,
                time_limit_s=time_limit,
                stop_event=stop_event,
            )

            t_elapsed = time.perf_counter() - t_start
            n_reloc = result["best"]
            if n_reloc < 0:
                n_reloc = int(env.get_metrics().get("relocations", 0.0))

            metrics: Dict = {
                "relocations": float(n_reloc),
                "steps": float(n_reloc),
                "time": float(n_reloc),
                "nodes_explored": float(result["nodes"]),
                "root_lower_bound": float(result["root_lb"]),
                "cache_hits": float(result["cache_hits"]),
                "pdb_hits": float(result["pdb_hits"]),
                "pdb_depth_built": float(pdb_built_depth),
                "optimal_proven": 1.0 if result["optimal"] else 0.0,
                "solve_time_s": round(t_elapsed, 4),
            }
            all_metrics.append(metrics)
            primary = float(n_reloc)

            if primary < self._best_metric:
                self._best_metric = primary
                self._best_solution = []  # move list not tracked (state-space search)

            self._push(
                result_queue,
                step=seed + 1,
                metric=primary,
                metrics=metrics,
                progress=(seed + 1) / n_seeds,
                snapshot=env.get_state_snapshot(),
            )

            print(
                f"[KuArthanari2016Abstraction] seed={seed}  relocations={n_reloc}  "
                f"optimal={result['optimal']}  nodes={result['nodes']}  "
                f"cache_hits={result['cache_hits']}  pdb_hits={result['pdb_hits']}  "
                f"t={t_elapsed:.3f}s",
                file=sys.stderr,
                flush=True,
            )

        if all_metrics:
            agg = {
                k: float(np.mean([m[k] for m in all_metrics if k in m]))
                for k in all_metrics[0]
            }
            self._push(
                result_queue,
                step=n_seeds,
                metric=self._best_metric,
                metrics=agg,
                progress=1.0,
            )

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    # ---------------------------------------------------------------- #
    # Config schema (for GUI / API)                                      #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "time_limit_s": {
                "type": "float", "default": 300.0, "min": 1.0, "max": 86400.0,
                "label": "Time limit (s)",
                "help": (
                    "Per-seed search time limit (excludes the one-off PDB "
                    "build, which is amortised across all seeds)."
                ),
            },
            "pdb_depth": {
                "type": "int", "default": 5, "min": 0, "max": 14,
                "label": "PDB depth (units)",
                "help": (
                    "Number of remaining containers for which the pattern "
                    "database (Section 4.3) stores exact optimal costs. "
                    "Reduction factor over the raw state space grows fast "
                    "with depth (paper's Table 3) but so does build time/"
                    "memory -- the paper notes ~10-15 units is the practical "
                    "ceiling; start low (3-6) and increase if build time "
                    "stays well under pdb_build_time_limit_s."
                ),
            },
            "cache_depth": {
                "type": "int", "default": 20, "min": 0, "max": 200,
                "label": "Forward cache depth (CC_n)",
                "help": (
                    "Number of levels from the root for which visited nodes "
                    "are cached by abstract state during the forward search "
                    "(Section 3.4's CC_n). Large values cache more of the "
                    "tree at the cost of memory."
                ),
            },
            "max_cache_size": {
                "type": "int", "default": 1_000_000, "min": 1_000, "max": 20_000_000,
                "label": "Max forward-cache entries",
                "help": (
                    "Upper bound on cached forward-search entries "
                    "(paper's Pmax_size, used at 20,000,000 in Java; "
                    "kept smaller by default for a Python process)."
                ),
            },
            "pdb_build_time_limit_s": {
                "type": "float", "default": 60.0, "min": 1.0, "max": 3600.0,
                "label": "PDB build time limit (s)",
                "help": (
                    "One-off wall-clock budget for building the pattern "
                    "database before the first seed's search starts. If "
                    "exceeded, the PDB is truncated to whatever depth "
                    "finished in time (search still runs correctly, just "
                    "without the deeper backward-search shortcut)."
                ),
            },
        })
        return base
