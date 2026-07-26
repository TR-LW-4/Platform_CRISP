"""
Lee & Chao (2009) neighborhood-search heuristic for export-container
pre-marshalling, with a solver-assisted (BIP) sequence reduction stage.

Paper:
Y. Lee, S.-L. Chao,
"A neighborhood search heuristic for pre-marshalling export containers",
EJOR 196 (2009) 468-475.
"""

from __future__ import annotations

import multiprocessing as mp
import random
import time as _time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from core.base_algorithm import AlgorithmConfig, BaseAlgorithm
from core.layout_trace import trace_layout
from .core import (
    Stacks,
    bay_mis_overlay_index,
    neighborhood_mutation,
    objective_value,
    random_empty_stack_prefix,
    reduce_misoverlay_suffix,
    reduce_sequence_rule,
    simulate_sequence,
)
from .solver_core import apply_merges, extract_merge_candidates, solve_best_merges


Move = Tuple[int, int]  # (src_idx, dst_idx)


def _encode_action(src_idx: int, dst_idx: int, n_stacks: int) -> int:
    # CRP_Prem packing: src*(n-1) + dst' where dst' skips src.
    dst_flat = dst_idx if dst_idx < src_idx else dst_idx - 1
    return src_idx * (n_stacks - 1) + dst_flat


class LeeChao2009NSIP(BaseAlgorithm):
    name = "Lee–Chao (2009) NS+IP"
    category = "Heuristic"
    description = (
        "[single-bay origin]  Neighborhood search (threshold accepting) + "
        "binary-IP sequence reduction for pre-marshalling. "
        "Solver-assisted heuristic (not an exact method)."
    )
    compatible_problems = ["CRP-Prem"]
    step_label = "Seed"
    requires_solver = True
    solver_backend = "gurobi"

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    def train(
        self,
        problem_factory: Callable,
        result_queue: mp.Queue,
        stop_event: mp.Event,
    ) -> None:
        cfg = self.config
        n_seeds = max(1, int(cfg.num_eval_seeds))
        all_metrics: List[Dict[str, float]] = []

        # Defaults follow the paper's spirit.
        max_outer = int(cfg.extra.get("max_outer_iterations", 30))
        ip_rounds = int(cfg.extra.get("ip_rounds_per_outer", 3))
        init_empty_runs = int(cfg.extra.get("init_empty_runs", 3))
        ta_init_ratio = float(cfg.extra.get("ta_init_ratio", 0.1))
        ta_decay = float(cfg.extra.get("ta_decay", 0.99))
        ta_max_steps = int(cfg.extra.get("ta_max_steps", 300))
        ta_decay_every = int(cfg.extra.get("ta_decay_every", 20))
        p_add = float(cfg.extra.get("p_add", 0.6))
        p_delete = float(cfg.extra.get("p_delete", 0.2))
        p_reloc = float(cfg.extra.get("p_relocate", 0.2))
        w_idx = float(cfg.extra.get("w_misoverlay", 1.0))
        w_len = float(cfg.extra.get("w_sequence_len", 0.1))
        ip_time_limit = float(cfg.extra.get("ip_time_limit", 5.0))
        suffix_max_moves = int(cfg.extra.get("suffix_max_moves", 64))

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            rng = random.Random(int(cfg.seed) + seed)
            trace_layout(
                f"LeeChao2009NSIP.train seed {seed+1}/{n_seeds} "
                f"outer={max_outer} ip_rounds={ip_rounds}"
            )

            env = problem_factory()
            env.config.seed = seed
            env.reset()

            stack_keys = list(env.yard.stacks.keys())
            n_stacks = len(stack_keys)
            key_to_idx = {k: i for i, k in enumerate(stack_keys)}
            max_tiers = int(env.config.max_tiers)

            stacks0: Stacks = {
                key_to_idx[k]: list(st.priority_snapshot())
                for k, st in env.yard.stacks.items()
            }

            seq: List[Move] = []

            # Initial diversification from paper's minor subroutine idea.
            for _ in range(max(0, init_empty_runs)):
                pre = random_empty_stack_prefix(stacks0, max_tiers, rng)
                seq.extend(pre)
                seq, _, _ = simulate_sequence(stacks0, seq, max_tiers)

            best_seq = list(seq)
            best_metric = float("inf")
            done_early = False

            for outer in range(1, max_outer + 1):
                if stop_event.is_set():
                    break

                seq, stacks_cur, _ = simulate_sequence(stacks0, seq, max_tiers)
                z_cur = objective_value(seq, stacks_cur, w_idx=w_idx, w_len=w_len)
                z_best = z_cur
                seq_best_local = list(seq)
                seq_work = list(seq)
                T = max(1.0, ta_init_ratio * max(1e-6, z_cur))
                non_improve = 0

                # Major subroutine 1: threshold-accepting neighborhood search.
                while T >= 1.0 and non_improve < ta_max_steps:
                    if stop_event.is_set():
                        break
                    cand = neighborhood_mutation(
                        seq_work, n_stacks, p_add, p_delete, p_reloc, rng
                    )
                    cand, stacks_cand, _ = simulate_sequence(stacks0, cand, max_tiers)
                    z = objective_value(cand, stacks_cand, w_idx=w_idx, w_len=w_len)
                    if z < z_best + T:
                        seq_work = cand
                        if z < z_best:
                            z_best = z
                            seq_best_local = list(cand)
                            non_improve = 0
                        else:
                            non_improve += 1
                    else:
                        non_improve += 1
                    if non_improve > 0 and non_improve % max(1, ta_decay_every) == 0:
                        T *= ta_decay

                seq = list(seq_best_local)

                # Major subroutine 2 + minor subroutines.
                for _ in range(max(0, ip_rounds)):
                    seq, stacks_cur, executed = simulate_sequence(stacks0, seq, max_tiers)

                    cands = extract_merge_candidates(executed)
                    if cands:
                        chosen = solve_best_merges(
                            n_moves=len(seq),
                            candidates=cands,
                            time_limit=ip_time_limit,
                        )
                        seq = apply_merges(seq, cands, chosen)

                    seq = reduce_sequence_rule(seq, stacks0, max_tiers)

                    seq_feas, stacks_after, _ = simulate_sequence(stacks0, seq, max_tiers)
                    suffix = reduce_misoverlay_suffix(
                        stacks_after, max_tiers, max_moves=suffix_max_moves
                    )
                    seq = seq_feas + suffix

                seq, stacks_final, _ = simulate_sequence(stacks0, seq, max_tiers)
                mis_idx = float(bay_mis_overlay_index(stacks_final))
                z_fin = objective_value(seq, stacks_final, w_idx=w_idx, w_len=w_len)

                if z_fin < best_metric:
                    best_metric = z_fin
                    best_seq = list(seq)

                self._push(
                    result_queue,
                    step=seed * max_outer + outer,
                    metric=z_fin,
                    metrics={
                        "objective": float(z_fin),
                        "moves": float(len(seq)),
                        "mis_overlay_index": mis_idx,
                        "bad_overlaps": mis_idx,
                        "time": float(len(seq)),
                        "seed": float(seed),
                        "outer_iter": float(outer),
                    },
                    progress=(seed * max_outer + outer) / max(1, n_seeds * max_outer),
                    snapshot=env.get_state_snapshot(),
                )

                if mis_idx <= 0.0:
                    done_early = True
                    break

            # Final seed metrics and action replay export.
            best_seq, best_stacks, _ = simulate_sequence(stacks0, best_seq, max_tiers)
            best_mis = float(bay_mis_overlay_index(best_stacks))
            best_moves = float(len(best_seq))
            metric = float(w_idx * best_mis + w_len * best_moves)

            actions = [_encode_action(s, d, n_stacks) for (s, d) in best_seq]
            if metric < self._best_metric:
                self._best_metric = metric
                self._best_solution = actions

            seed_metrics = {
                "objective": metric,
                "moves": best_moves,
                "mis_overlay_index": best_mis,
                "bad_overlaps": best_mis,
                "time": best_moves,
                "solved": 1.0 if best_mis <= 0.0 else 0.0,
            }
            all_metrics.append(seed_metrics)

            self._push(
                result_queue,
                step=seed + 1,
                metric=metric,
                metrics=seed_metrics,
                progress=(seed + 1) / n_seeds,
                snapshot=env.get_state_snapshot(),
                extra={"terminated_early": bool(done_early)},
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

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "num_eval_seeds": {
                "type": "int", "default": 10, "min": 1, "max": 100,
                "label": "Evaluation seeds",
            },
            "max_outer_iterations": {
                "type": "int", "default": 30, "min": 1, "max": 500,
                "label": "Outer iterations",
            },
            "init_empty_runs": {
                "type": "int", "default": 3, "min": 0, "max": 20,
                "label": "Initial stack-emptying runs",
            },
            "ip_rounds_per_outer": {
                "type": "int", "default": 3, "min": 0, "max": 20,
                "label": "IP reductions per outer iteration",
            },
            "ip_time_limit": {
                "type": "float", "default": 5.0, "min": 0.1, "max": 120.0,
                "label": "Gurobi time limit per IP call (s)",
            },
            "ta_init_ratio": {
                "type": "float", "default": 0.1, "min": 0.001, "max": 1.0,
                "label": "TA initial threshold ratio",
            },
            "ta_decay": {
                "type": "float", "default": 0.99, "min": 0.8, "max": 0.999,
                "label": "TA threshold decay",
            },
            "ta_max_steps": {
                "type": "int", "default": 300, "min": 10, "max": 5000,
                "label": "TA max inner steps",
            },
            "p_add": {
                "type": "float", "default": 0.6, "min": 0.0, "max": 1.0,
                "label": "Neighborhood prob: add move",
            },
            "p_delete": {
                "type": "float", "default": 0.2, "min": 0.0, "max": 1.0,
                "label": "Neighborhood prob: delete move",
            },
            "p_relocate": {
                "type": "float", "default": 0.2, "min": 0.0, "max": 1.0,
                "label": "Neighborhood prob: relocate move",
            },
            "w_misoverlay": {
                "type": "float", "default": 1.0, "min": 0.0, "max": 100.0,
                "label": "Objective weight: mis-overlay index",
            },
            "w_sequence_len": {
                "type": "float", "default": 0.1, "min": 0.0, "max": 100.0,
                "label": "Objective weight: sequence length",
            },
            "suffix_max_moves": {
                "type": "int", "default": 64, "min": 0, "max": 1000,
                "label": "Minor reducer max appended moves",
            },
        })
        return base

