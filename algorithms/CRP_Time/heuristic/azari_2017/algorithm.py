"""
Azari2017CSUM
<2017> <heuristic> <time> <single-bay> <CRP-Time>
CSUM tree search minimizing crane working time
time_limit_s --- 5 --- Search time limit (s)
max_branches_b --- 6 --- Constant-summation branch width

------------------------------- Reference --------------------------------
E. Azari, H. Eskandari, A. Nourmohammadi,
"Decreasing the crane working time in retrieving the containers from a bay",
Scientia Iranica E 24 (2017) 309–318.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
import multiprocessing as mp
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.base_algorithm import AlgorithmConfig, BaseAlgorithm
from core.objectives import (
    KinematicsModel,
    ObjectiveSpec,
    evaluate_plan_objectives,
    movement_objective_cost,
)
from core.plan import Movement, RelocationPlan


_EPS = 1e-9


@dataclass
class _SearchResult:
    moves: List[Movement]
    objective_value: float
    total_moves: int
    relocations: int


class _CSUMSolver:
    """Depth-first tree search implementing GBH + CSUM branching."""

    def __init__(
        self,
        stacks: Sequence[Sequence[int]],
        positions: Sequence[Tuple[int, int]],
        max_tiers: int,
        kinematics: KinematicsModel,
        objective_spec: ObjectiveSpec,
        time_limit_s: float,
        max_branches_b: int,
    ) -> None:
        self.initial_stacks: List[List[int]] = [list(s) for s in stacks]
        self.positions: List[Tuple[int, int]] = list(positions)
        self.max_tiers = int(max_tiers)
        self.kin = kinematics
        self.objective_spec = objective_spec

        self.time_limit_s = float(max(0.1, time_limit_s))
        self.max_branches_b = int(max(1, max_branches_b))
        self.start_ts = time.monotonic()

        self.targets: List[int] = sorted(
            p for st in self.initial_stacks for p in st
        )
        self.empty_lowest = (max(self.targets) + 1) if self.targets else 1

        self.best_time = float("inf")
        self.best_total_moves = float("inf")
        self.best_moves: List[Movement] = []

        self._set_a: set[int] = set()

    def solve(self) -> _SearchResult:
        if not self.targets:
            return _SearchResult([], 0.0, 0, 0)

        initial_moves, complete = self._run_gbh_initial()
        if complete:
            self._accept_solution_if_better(initial_moves)

        reloc_count = Counter(
            m.container_id for m in initial_moves if not m.is_retrieval
        )
        self._set_a = {cid for cid, cnt in reloc_count.items() if cnt > 1}

        stacks = copy.deepcopy(self.initial_stacks)
        path: List[Movement] = []
        self._dfs(
            stacks=stacks,
            target_idx=0,
            crane_pos=(1, 1),
            nmov=0,
            tcw=0.0,
            path=path,
        )

        relocs = sum(1 for m in self.best_moves if not m.is_retrieval)
        return _SearchResult(
            moves=self.best_moves[:],
            objective_value=float(self.best_time),
            total_moves=int(self.best_total_moves),
            relocations=int(relocs),
        )

    def _timed_out(self) -> bool:
        return (time.monotonic() - self.start_ts) >= self.time_limit_s

    def _lowest(self, stack: Sequence[int]) -> int:
        return min(stack) if stack else self.empty_lowest

    def _is_bg(self, q: int, dst_stack: Sequence[int]) -> bool:
        return q < self._lowest(dst_stack)

    def _is_bb(self, q: int, dst_stack: Sequence[int]) -> bool:
        return q > self._lowest(dst_stack)

    def _find_target_stack(self, stacks: Sequence[Sequence[int]], target: int) -> int:
        for i, st in enumerate(stacks):
            if target in st:
                return i
        return -1

    def _lower_bound_moves_2x_plus_y(self, stacks: Sequence[Sequence[int]]) -> int:
        x_bad = 0
        y_good = 0
        for st in stacks:
            for i, c in enumerate(st):
                bad = False
                for j in range(i):
                    if st[j] < c:
                        bad = True
                        break
                if bad:
                    x_bad += 1
                else:
                    y_good += 1
        return 2 * x_bad + y_good

    def _apply_relocation(
        self,
        stacks: List[List[int]],
        src: int,
        dst: int,
    ) -> int:
        q = stacks[src].pop()
        stacks[dst].append(q)
        return q

    def _undo_relocation(
        self,
        stacks: List[List[int]],
        src: int,
        dst: int,
        q: int,
    ) -> None:
        popped = stacks[dst].pop()
        if popped != q:
            raise RuntimeError("CSUM rollback mismatch on relocation")
        stacks[src].append(q)

    def _apply_retrieval(
        self,
        stacks: List[List[int]],
        src: int,
        target: int,
    ) -> None:
        popped = stacks[src].pop()
        if popped != target:
            raise RuntimeError("CSUM retrieval mismatch")

    def _undo_retrieval(
        self,
        stacks: List[List[int]],
        src: int,
        target: int,
    ) -> None:
        stacks[src].append(target)

    def _true_eligible_dsts(self, stacks: Sequence[Sequence[int]], src: int) -> List[int]:
        return [
            i for i, st in enumerate(stacks)
            if i != src and len(st) < self.max_tiers
        ]

    def _best_bg_stack(
        self,
        q: int,
        stacks: Sequence[Sequence[int]],
        candidates: Sequence[int],
    ) -> Optional[int]:
        bg = [i for i in candidates if self._is_bg(q, stacks[i])]
        if not bg:
            return None
        return min(bg, key=lambda i: (self._lowest(stacks[i]), i))

    def _best_bb_stack(
        self,
        q: int,
        stacks: Sequence[Sequence[int]],
        candidates: Sequence[int],
    ) -> Optional[int]:
        bb = [i for i in candidates if self._is_bb(q, stacks[i])]
        if not bb:
            return None
        return max(bb, key=lambda i: (self._lowest(stacks[i]), -i))

    def _gbh_candidates(
        self,
        q: int,
        src: int,
        stacks: Sequence[Sequence[int]],
    ) -> List[int]:
        candidates = self._true_eligible_dsts(stacks, src)
        if not candidates:
            return []

        best_bg = self._best_bg_stack(q, stacks, candidates)
        if best_bg is not None:
            out = [best_bg]
            best_bb = self._best_bb_stack(q, stacks, candidates)
            if best_bb is not None and best_bb != best_bg:
                out.append(best_bb)
            return out

        bb_sorted = sorted(
            [i for i in candidates if self._is_bb(q, stacks[i])],
            key=lambda i: (self._lowest(stacks[i]), -i),
            reverse=True,
        )
        return bb_sorted[:2]

    def _csum_b_candidates(
        self,
        q: int,
        src: int,
        stacks: Sequence[Sequence[int]],
    ) -> List[int]:
        n = len(stacks)
        lows = [self._lowest(st) for st in stacks]

        out: List[int] = []

        left = [
            i for i in range(0, src)
            if len(stacks[i]) < self.max_tiers and self._is_bg(q, stacks[i])
        ]
        if left:
            best_left = min(left, key=lambda i: (lows[i], i))
            out.append(best_left)

        for m in range(src + 1, n):
            if len(stacks[m]) >= self.max_tiers:
                continue
            if not self._is_bg(q, stacks[m]):
                continue
            prev = [i for i in range(0, m) if i != src]
            if all(lows[m] < lows[i] for i in prev):
                out.append(m)

        if not out:
            return []

        dedup: List[int] = []
        seen = set()
        for i in out:
            if i not in seen:
                seen.add(i)
                dedup.append(i)
        return dedup[: self.max_branches_b]

    def _accept_solution_if_better(self, moves: Sequence[Movement]) -> None:
        tcw = 0.0
        crane_pos = (1, 1)
        for mv in moves:
            dt, _time_cost, crane_pos = movement_objective_cost(
                mv, self.objective_spec, self.kin, crane_pos
            )
            tcw += dt
        n_total = len(moves)
        better = (tcw + _EPS < self.best_time) or (
            abs(tcw - self.best_time) <= _EPS and n_total < self.best_total_moves
        )
        if better:
            self.best_time = float(tcw)
            self.best_total_moves = int(n_total)
            self.best_moves = list(moves)

    def _run_gbh_initial(self) -> Tuple[List[Movement], bool]:
        stacks = copy.deepcopy(self.initial_stacks)
        target_idx = 0
        moves: List[Movement] = []

        while target_idx < len(self.targets):
            target = self.targets[target_idx]
            src = self._find_target_stack(stacks, target)
            if src < 0:
                return moves, False

            if stacks[src] and stacks[src][-1] == target:
                mv = Movement(
                    target,
                    self.positions[src],
                    None,
                    from_tier=len(stacks[src]),
                )
                moves.append(mv)
                self._apply_retrieval(stacks, src, target)
                target_idx += 1
                continue

            if not stacks[src]:
                break

            q = stacks[src][-1]
            cand = self._gbh_candidates(q, src, stacks)
            if not cand:
                return moves, False

            dst = cand[0]
            mv = Movement(
                q,
                self.positions[src],
                self.positions[dst],
                from_tier=len(stacks[src]),
                to_tier=len(stacks[dst]) + 1,
            )
            self._apply_relocation(stacks, src, dst)
            moves.append(mv)

        return moves, True

    def _dfs(
        self,
        stacks: List[List[int]],
        target_idx: int,
        crane_pos: Tuple[int, int],
        nmov: int,
        tcw: float,
        path: List[Movement],
    ) -> None:
        if self._timed_out():
            return

        if tcw >= self.best_time - _EPS:
            return

        if target_idx >= len(self.targets):
            better = (tcw + _EPS < self.best_time) or (
                abs(tcw - self.best_time) <= _EPS and nmov < self.best_total_moves
            )
            if better:
                self.best_time = float(tcw)
                self.best_total_moves = int(nmov)
                self.best_moves = path[:]
            return

        lb = self._lower_bound_moves_2x_plus_y(stacks)
        if (
            self.objective_spec.mode == "relocations"
            and nmov + lb > self.best_total_moves
        ):
            return

        target = self.targets[target_idx]
        src = self._find_target_stack(stacks, target)
        if src < 0:
            return

        if stacks[src] and stacks[src][-1] == target:
            mv = Movement(
                target,
                self.positions[src],
                None,
                from_tier=len(stacks[src]),
            )
            dt, _time_cost, next_pos = movement_objective_cost(
                mv, self.objective_spec, self.kin, crane_pos
            )
            new_t = tcw + dt
            if new_t >= self.best_time - _EPS:
                return

            self._apply_retrieval(stacks, src, target)
            path.append(mv)
            self._dfs(
                stacks=stacks,
                target_idx=target_idx + 1,
                crane_pos=next_pos,
                nmov=nmov + 1,
                tcw=new_t,
                path=path,
            )
            path.pop()
            self._undo_retrieval(stacks, src, target)
            return

        q = stacks[src][-1]
        if q in self._set_a:
            candidates = self._gbh_candidates(q, src, stacks)
        else:
            candidates = self._csum_b_candidates(q, src, stacks)
            if not candidates:
                candidates = self._gbh_candidates(q, src, stacks)

        if not candidates:
            return

        for dst in candidates:
            if self._timed_out():
                return
            mv = Movement(
                stacks[src][-1],
                self.positions[src],
                self.positions[dst],
                from_tier=len(stacks[src]),
                to_tier=len(stacks[dst]) + 1,
            )
            moved_q = self._apply_relocation(stacks, src, dst)
            dt, _time_cost, next_pos = movement_objective_cost(
                mv, self.objective_spec, self.kin, crane_pos
            )
            new_t = tcw + dt
            if new_t < self.best_time - _EPS:
                path.append(mv)
                self._dfs(
                    stacks=stacks,
                    target_idx=target_idx,
                    crane_pos=next_pos,
                    nmov=nmov + 1,
                    tcw=new_t,
                    path=path,
                )
                path.pop()
            self._undo_relocation(stacks, src, dst, moved_q)


class Azari2017CSUM(BaseAlgorithm):

    name = "Azari–Eskandari–Nourmohammadi (2017) CSUM"
    category = "Heuristic"
    description = (
        "[single-bay origin] CSUM heuristic for CRP-Time (Scientia Iranica 2017). "
        "Pipeline: GBH initial solution -> A/B partition by relocation count -> "
        "CSUM tree search minimizing crane working time (TCW). "
        "A-set containers follow GBH branching; B-set containers follow "
        "constant-summation branching. "
        "Implementation is self-contained and does not depend on other algorithms."
    )
    compatible_problems = ["CRP-Time"]
    geometry = "single-bay"
    objectives = ["crane_time"]
    fidelity = "faithful"
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
            initial_yard = copy.deepcopy(env.yard)

            positions = sorted(env.yard.stacks.keys())
            stacks = [
                [int(c.priority) for c in env.yard.stacks[pos].containers]
                for pos in positions
            ]

            kin = KinematicsModel.from_config_extra(env.config.extra)
            objective_spec = ObjectiveSpec.from_config(env.config)
            solver = _CSUMSolver(
                stacks=stacks,
                positions=positions,
                max_tiers=int(env.config.max_tiers),
                kinematics=kin,
                objective_spec=objective_spec,
                time_limit_s=float(cfg.extra.get("time_limit_s", 5.0)),
                max_branches_b=int(cfg.extra.get("max_branches_b", 6)),
            )
            result = solver.solve()

            actions = _moves_to_actions(result.moves, int(env.config.num_rows))
            metrics = evaluate_plan_objectives(
                RelocationPlan(list(result.moves)),
                objective_spec,
                kinematics=kin,
                initial_yard=initial_yard,
            )
            metrics["steps"] = float(result.total_moves)
            all_metrics.append(metrics)

            primary = float(metrics["objective_value"])
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
            "time_limit_s": {
                "type": "float", "default": 5.0, "min": 0.1, "max": 120.0,
                "label": "Search time limit (s)",
                "help": "Wall-clock limit for each seed's GBH+CSUM search.",
            },
            "max_branches_b": {
                "type": "int", "default": 6, "min": 1, "max": 32,
                "label": "Max B-set branches",
                "help": "Maximum CSUM destination branches explored for B-set containers.",
            },
        })
        return base


def _moves_to_actions(moves: Sequence[Movement], num_rows: int) -> List[int]:
    actions: List[int] = []
    for mv in moves:
        if mv.to_pos is None:
            continue
        bay, row = mv.to_pos
        actions.append((bay - 1) * num_rows + (row - 1))
    return actions
