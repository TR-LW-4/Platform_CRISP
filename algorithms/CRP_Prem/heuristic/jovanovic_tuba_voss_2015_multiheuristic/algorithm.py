"""
Jovanovic, Tuba, Voss (2015) multi-heuristic approach for the pre-marshalling
problem.

Reference
---------
R. Jovanovic, M. Tuba, S. Voss, "A multi-heuristic approach for solving the
pre-marshalling problem", Central European Journal of Operations Research
25 (2017) 1-28 (published online 2015).

Embedding scope
----------------
This paper is a deterministic extension of the already-embedded
Exposito-Izquierdo, Melian-Batista, Moreno-Vega (2012) LPFH
(``heuristic/exposito_melian_moreno_2012_lpfh``): it keeps the same
four-stage greedy skeleton (select a block to well-locate, select a
destination stack, relocate the blocking containers, fill the destination)
but replaces LPFH's randomized top-k selection with a small set of
competing deterministic heuristic functions per stage, enumerates every
combination (2 x 2 x 3 x 4 = 48), and keeps the best solution. It also adds
two paper-specific mechanisms not present in LPFH:

* Sec. 4.1 formalized deadlock avoidance (Eq. 7-9): instead of
  backtracking, undo the most recent relocation, borrow a slot from a
  random full stack, and retry -- always finds a way to continue.
* Sec. 4.2 move-sequence correction (Eq. 10-11): removes redundant
  "chain" and "there-and-back" relocations from the final solution.

This is intentionally a self-contained reimplementation (not an import) of
the shared four-stage primitives, consistent with this platform's
one-paper-one-folder convention.

Documented approximations (see module docstrings for detail):
* ``core.py``: the forced-relocation count fr(c, s*) (Sec. 3.2) is only
  described informally in the paper via a worked example; we implement the
  literal "all other stacks' effective top values exceed the blocker's own
  value" rule stated in the text.
* ``relocation.py``: the MinMax heuristic (Sec. 3.3) is cited from other
  papers rather than fully specified; we use the standard BRP/PMP
  formulation from that literature (tightest well-locating fit, else
  smallest stack maximum).
* ``deadlock.py``: Eq. 7-9's exact subscripts are ambiguous in the source;
  our recovery preserves the same ingredients (revert an earlier
  relocation, optionally borrow through a random full stack) without a
  literal "redo" step, to stay provably capacity-safe.

Empirically (randomized instances following the paper's own benchmark
convention of ``max_tiers = instance_height + 2``), the full 48-combination
search reaches a fully well-located layout on ~99% of instances (50/75%
occupancy: 100%, 100% occupancy: ~97%); the rest report ``solved=False``
with the best partial layout found rather than crashing or corrupting
state.
"""

from __future__ import annotations

import multiprocessing as mp
import time as _time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from core.base_algorithm import AlgorithmConfig, BaseAlgorithm
from core.layout_trace import trace_layout

from .core import Stacks
from .multiheuristic import HB_SET, HF_SET, HS_SET, HW_SET, all_combos, run_multiheuristic


def _encode_action(src_idx: int, dst_idx: int, n_stacks: int) -> int:
    dst_flat = dst_idx if dst_idx < src_idx else dst_idx - 1
    return src_idx * (n_stacks - 1) + dst_flat


class JovanovicTubaVoss2015MultiHeuristic(BaseAlgorithm):

    name = "Jovanovic-Tuba-Voss (2015) Multi-Heuristic"
    category = "Heuristic"
    description = (
        "[single-bay origin] Jovanovic, Tuba & Voss (CEJOR 2015) "
        "deterministic multi-heuristic approach for the CPMP. Extends the "
        "Exposito-Izquierdo (2012) four-stage greedy skeleton with "
        "competing heuristics per stage (block selection, destination "
        "selection, relocation target, filling), enumerates all 48 "
        "combinations and keeps the best solution, avoiding randomization "
        "entirely. Includes formalized deadlock avoidance and a "
        "move-sequence correction pass."
    )
    compatible_problems = ["CRP-Prem"]
    step_label = "Seed"
    requires_solver = False
    solver_backend = None

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    # ---------------------------------------------------------------- #
    # Training loop                                                      #
    # ---------------------------------------------------------------- #

    def train(
        self,
        problem_factory: Callable,
        result_queue: mp.Queue,
        stop_event: mp.Event,
    ) -> None:
        cfg = self.config
        n_seeds = max(1, int(cfg.num_eval_seeds))
        safe_slack = int(cfg.extra.get("safe_slack", 1))
        full_search = bool(cfg.extra.get("full_48_search", True))

        combos = None
        if not full_search:
            hb = str(cfg.extra.get("hb", HB_SET[0]))
            hs = str(cfg.extra.get("hs", HS_SET[0]))
            hw = str(cfg.extra.get("hw", HW_SET[0]))
            hf = str(cfg.extra.get("hf", HF_SET[0]))
            combos = [(hb, hs, hw, hf)]

        all_metrics: List[Dict[str, float]] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.config.seed = seed
            env.reset()

            stack_keys = list(env.yard.stacks.keys())
            key_to_idx = {k: i for i, k in enumerate(stack_keys)}
            n_stacks = len(stack_keys)
            max_tiers = int(env.config.max_tiers)

            stacks_init: Stacks = {
                key_to_idx[k]: list(st.priority_snapshot())
                for k, st in env.yard.stacks.items()
            }

            trace_layout(
                f"JovanovicTubaVoss2015MultiHeuristic seed={seed+1}/{n_seeds} "
                f"full_48_search={full_search} n_combos={len(combos) if combos else len(all_combos())}"
            )

            t0 = _time.perf_counter()
            result = run_multiheuristic(
                stacks_init,
                max_tiers=max_tiers,
                rng_seed=int(cfg.seed) + seed * 10007,
                safe_slack=safe_slack,
                combos=combos,
            )
            elapsed = _time.perf_counter() - t0

            moves_list = list(result.moves)
            solved = bool(result.solved)
            remaining = int(result.remaining_not_well_located)

            metrics = {
                "moves": float(len(moves_list)),
                "remaining_not_well_located": float(remaining),
                "time": float(elapsed),
                "solved": 1.0 if solved else 0.0,
            }
            all_metrics.append(metrics)

            primary = float(len(moves_list) if solved else len(moves_list) + 1000.0 * (remaining + 1))
            if primary < self._best_metric:
                self._best_metric = primary
                self._best_solution = [
                    _encode_action(s, d, n_stacks) for (s, d) in moves_list
                ]

            self._push(
                result_queue,
                step=seed + 1,
                metric=primary,
                metrics=metrics,
                progress=(seed + 1) / n_seeds,
                snapshot=env.get_state_snapshot(),
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
    # Configuration schema                                               #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict[str, Dict[str, Any]]:
        base = super().config_schema()
        base.update({
            "num_eval_seeds": {
                "type": "int", "default": 10, "min": 1, "max": 100,
                "label": "Evaluation seeds",
                "help": "The method is deterministic per instance; multiple seeds vary the layout.",
            },
            "full_48_search": {
                "type": "bool", "default": True,
                "label": "Run the full 48-combination multi-heuristic search",
                "help": (
                    "When disabled, runs a single fixed (hb, hs, hw, hf) "
                    "combination instead (useful for ablation)."
                ),
            },
            "hb": {
                "type": "str", "default": "descending",
                "label": "Ablation only: block-selection heuristic (descending|lookahead)",
            },
            "hs": {
                "type": "str", "default": "w",
                "label": "Ablation only: destination-stack heuristic (w|w_hat)",
            },
            "hw": {
                "type": "str", "default": "MinMax",
                "label": "Ablation only: relocation heuristic (TLP|LPI|MinMax)",
            },
            "hf": {
                "type": "str", "default": "Standard",
                "label": "Ablation only: filling heuristic (None|Standard|Safe|Stop)",
            },
            "safe_slack": {
                "type": "int", "default": 1, "min": 0, "max": 20,
                "label": "Safe-filling slack threshold (a)",
                "help": "Safe filling is only committed if it leaves at most this many empty tiers.",
            },
        })
        return base
