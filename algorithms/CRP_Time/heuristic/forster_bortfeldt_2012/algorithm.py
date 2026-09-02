"""
ForsterBortfeldt2012Retrieval
<2012> <heuristic> <time> <multi-bay> <CRP-Time>
Per-bay tree search merged by global lowest-group retrieval
per_bay_time_limit_s --- 3 --- Time limit per bay (s)
n_succ --- 5 --- Successor compound moves per node

------------------------------- Reference --------------------------------
F. Forster, A. Bortfeldt,
"A tree search heuristic for the container retrieval problem",
Operations Research Proceedings 2011, Springer, 2012.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from algorithms.CRP_D.heuristic.forster_bortfeldt_2012.layout import FBLayout
from algorithms.CRP_D.heuristic.forster_bortfeldt_2012.tree_search import run_tree_search
from core.base_algorithm import AlgorithmConfig, BaseAlgorithm
from core.objectives import KinematicsModel, compute_crane_time
from core.plan import Movement, RelocationPlan


Move = Tuple  # ('rem', s) | ('rel', d, r)
_StackItem = Tuple[int, int]  # (container_id, priority)


@dataclass
class _MergedResult:
    plan: RelocationPlan
    total_moves: int
    relocations: int


def _moves_to_actions(plan: RelocationPlan, num_rows: int) -> List[int]:
    actions: List[int] = []
    for m in plan.movements:
        if m.to_pos is None:
            continue
        bay, row = m.to_pos
        actions.append((bay - 1) * num_rows + (row - 1))
    return actions


def _build_single_bay_layout(
    bay_stacks: List[List[_StackItem]],
    max_tiers: int,
) -> FBLayout:
    pri_stacks = [[p for (_cid, p) in st] for st in bay_stacks]
    all_p = [p for st in pri_stacks for p in st]
    g_max = max(all_p) if all_p else 1
    n = len(all_p)
    next_group = min(all_p) if all_p else g_max + 1
    return FBLayout(
        stacks=[list(st) for st in pri_stacks],
        S=len(pri_stacks),
        H=max_tiers,
        G=g_max,
        n=n,
        next_group=next_group,
    )


def _extract_block_state(env) -> Dict[int, List[List[_StackItem]]]:
    by_bay: Dict[int, List[List[_StackItem]]] = {}
    for bay in range(1, env.config.num_bays + 1):
        rows: List[List[_StackItem]] = []
        for row in range(1, env.config.num_rows + 1):
            stack = env.yard.stacks[(bay, row)]
            rows.append([(int(c.id), int(c.priority)) for c in stack.containers])
        by_bay[bay] = rows
    return by_bay


def _lowest_group_in_bay(bay_stacks: List[List[_StackItem]]) -> Optional[int]:
    vals = [p for st in bay_stacks for (_cid, p) in st]
    return min(vals) if vals else None


def _lowest_group_in_block(block: Dict[int, List[List[_StackItem]]]) -> Optional[int]:
    vals: List[int] = []
    for bay_stacks in block.values():
        vals.extend([p for st in bay_stacks for (_cid, p) in st])
    return min(vals) if vals else None


def _solve_each_bay(
    block: Dict[int, List[List[_StackItem]]],
    max_tiers: int,
    per_bay_time_limit_s: float,
    n_succ: int,
    cm_stop_threshold: int,
    max_flg_bb: int,
    max_gg: int,
) -> Dict[int, List[Move]]:
    bay_ops: Dict[int, List[Move]] = {}
    for bay, bay_stacks in block.items():
        layout = _build_single_bay_layout(bay_stacks, max_tiers=max_tiers)
        if layout.is_empty():
            bay_ops[bay] = []
            continue
        ops, _ = run_tree_search(
            layout=layout,
            time_limit=float(per_bay_time_limit_s),
            n_succ=int(n_succ),
            cm_stop_threshold=int(cm_stop_threshold),
            max_flg_bb=int(max_flg_bb),
            max_gg=int(max_gg),
        )
        bay_ops[bay] = list(ops)
    return bay_ops


def _merge_bay_solutions(
    block: Dict[int, List[List[_StackItem]]],
    bay_ops: Dict[int, List[Move]],
) -> _MergedResult:
    pointers = {bay: 0 for bay in bay_ops}
    plan = RelocationPlan()

    while True:
        lowest_block = _lowest_group_in_block(block)
        if lowest_block is None:
            break

        progress_in_round = False
        for bay in sorted(block.keys()):
            bay_lowest = _lowest_group_in_bay(block[bay])
            if bay_lowest is None or bay_lowest != lowest_block:
                continue

            ops = bay_ops.get(bay, [])
            ptr = pointers.get(bay, 0)
            removed_target = False

            while ptr < len(ops) and not removed_target:
                op = ops[ptr]
                ptr += 1
                progress_in_round = True

                if op[0] == "rel":
                    src = int(op[1])
                    dst = int(op[2])
                    if not block[bay][src]:
                        continue
                    item = block[bay][src].pop()
                    block[bay][dst].append(item)
                    cid, _pri = item
                    plan.add(Movement(cid, (bay, src + 1), (bay, dst + 1)))
                elif op[0] == "rem":
                    src = int(op[1])
                    if not block[bay][src]:
                        continue
                    cid, pri = block[bay][src].pop()
                    plan.add(Movement(cid, (bay, src + 1), None))
                    if pri == lowest_block:
                        removed_target = True

            pointers[bay] = ptr

        if not progress_in_round:
            # No remaining applicable moves while containers still exist.
            # Abort merge defensively to avoid infinite loop.
            break

    n_reloc = plan.num_relocations()
    return _MergedResult(
        plan=plan,
        total_moves=plan.num_moves(),
        relocations=n_reloc,
    )


class ForsterBortfeldt2012Retrieval(BaseAlgorithm):
    """
    Forster-Bortfeldt tree-search adaptation for CRP-Time.
    """

    name = "Forster–Bortfeldt (2012) Retrieval Tree Search"
    category = "Heuristic"
    description = (
        "[native multi-bay] OR Proceedings 2012 adaptation: "
        "solve bays independently with single-bay tree search, then merge bay "
        "solutions by repeatedly executing moves until the block's global lowest "
        "group is removed. Objective: crane working time."
    )
    compatible_problems = ["CRP-Time"]
    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    def train(
        self,
        problem_factory: Callable,
        result_queue: mp.Queue,
        stop_event: mp.Event,
    ) -> None:
        cfg = self.config
        n_seeds = 1  # multi-seed eval removed; single run only
        all_metrics: List[Dict[str, float]] = []

        for seed_idx in range(n_seeds):
            if stop_event.is_set():
                break

            env = problem_factory()
            env.config.seed = int(cfg.seed) + seed_idx
            env.reset()

            block_state = _extract_block_state(env)
            bay_ops = _solve_each_bay(
                block=block_state,
                max_tiers=int(env.config.max_tiers),
                per_bay_time_limit_s=float(cfg.extra.get("per_bay_time_limit_s", 3.0)),
                n_succ=int(cfg.extra.get("n_succ", 5)),
                cm_stop_threshold=int(cfg.extra.get("cm_stop_threshold", 150)),
                max_flg_bb=int(cfg.extra.get("max_flg_bb", 3)),
                max_gg=int(cfg.extra.get("max_gg", 2)),
            )
            merged = _merge_bay_solutions(block_state, bay_ops)

            kin = KinematicsModel.from_config_extra(env.config.extra)
            crane_time = float(compute_crane_time(merged.plan, kin))

            actions = _moves_to_actions(merged.plan, int(env.config.num_rows))
            replay_metrics = env.evaluate(actions)
            metrics = {
                "time": float(replay_metrics.get("time", crane_time)),
                "crane_time": float(replay_metrics.get("crane_time", crane_time)),
                "relocations": float(replay_metrics.get("relocations", merged.relocations)),
                "steps": float(replay_metrics.get("steps", merged.total_moves)),
            }
            all_metrics.append(metrics)

            primary = float(metrics["crane_time"])
            if primary < self._best_metric:
                self._best_metric = primary
                self._best_solution = actions

            self._push(
                result_queue,
                step=seed_idx + 1,
                metric=primary,
                metrics=metrics,
                progress=(seed_idx + 1) / n_seeds,
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

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "per_bay_time_limit_s": {
                "type": "float", "default": 3.0, "min": 0.1, "max": 120.0,
                "label": "Per-bay solve time limit (s)",
                "help": "Time budget for each independent bay tree search.",
            },
            "n_succ": {
                "type": "int", "default": 5, "min": 1, "max": 20,
                "label": "Max successors (nSucc)",
                "help": "Branching factor in single-bay tree search.",
            },
            "cm_stop_threshold": {
                "type": "int", "default": 150, "min": 10, "max": 1000,
                "label": "cmStopThreshold",
                "help": "Compound-move depth controller in single-bay search.",
            },
            "max_flg_bb": {
                "type": "int", "default": 3, "min": 1, "max": 10,
                "label": "Max FLG_BB moves",
                "help": "Max FLG_BB productive moves per search step.",
            },
            "max_gg": {
                "type": "int", "default": 2, "min": 0, "max": 10,
                "label": "Max GG moves",
                "help": "Max GG productive moves per search step.",
            },
        })
        return base
