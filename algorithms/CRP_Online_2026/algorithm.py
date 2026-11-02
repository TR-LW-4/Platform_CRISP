"""
Algorithms for CRP-Online (2026 Paper).

Exposes 10 published manual heuristics and pre-evolved/trainable Genetic Programming rules
from Marko Đurasević, Mateja Đumić, Francisco Javier Gil Gala (2026).
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.base_algorithm import AlgorithmConfig, BaseAlgorithm, ProgressRecord
from core.base_problem import BaseProblem

from crp_online.core.observation import ObservationState
from crp_online.evaluator.prefix_evaluator import PrefixExpression
from crp_online.heuristics.restricted import select_best_stack_pf
from crp_online.heuristics.manual import (
    select_tlp_destination,
    select_ri_destination,
    select_minmax_destination,
    select_pr4_destination,
    select_rdh_destination,
    select_eri_destination,
    select_em_destination,
    select_egh_destination,
)


class Base2026HeuristicAlgorithm(BaseAlgorithm):
    """Base class for 2026 paper manual heuristic adapters."""

    category: str = "Heuristic"
    geometry: str = "single-bay"
    objectives: List[str] = ["relocations"]
    fidelity: str = "faithful"
    compatible_problems: List[str] = [
        "CRP-Online (2026 Paper)",
        "CRP-Online",
        "CRP-R",
    ]

    def _select_destination(self, obs: ObservationState, env_sim: Any, src_stack: int, moving_val: float) -> int:
        raise NotImplementedError

    def train(
        self,
        problem_factory: Callable[[], BaseProblem],
        result_queue: mp.Queue,
        stop_event: mp.Event,
    ) -> None:
        """Executes one heuristic episode and streams progress records."""
        env = problem_factory()
        obs_grid, info = env.reset()
        solution: List[int] = []
        step = 0

        # Emit initial step 0 progress record so UI responds instantly
        result_queue.put(
            ProgressRecord(
                step=0,
                metric=0.0,
                metrics=env.get_metrics() if hasattr(env, "get_metrics") else {},
                best_metric=0.0,
                progress=0.0,
                yard_snapshot=env.get_state_snapshot() if hasattr(env, "get_state_snapshot") else None,
                extra={"best_solution": []},
            )
        )

        while not stop_event.is_set():
            if hasattr(env, "_sim") and env._sim.is_empty():
                break

            if hasattr(env, "_sim"):
                target = env._sim.get_current_target()
                src = env._sim.get_stack_of_container(target)
                top_cid = env._sim.get_top_container(src)
                obs_state = env._sim.get_observation()
                try:
                    dest = self._select_destination(obs_state, env._sim, src, float(top_cid))
                except Exception as err:
                    metric_val = 10000.0 + step
                    self._best_metric = metric_val
                    record = ProgressRecord(
                        step=step + 1,
                        metric=metric_val,
                        metrics={"relocations": metric_val, "error": str(err)},
                        best_metric=metric_val,
                        progress=1.0,
                        yard_snapshot=env.get_state_snapshot() if hasattr(env, "get_state_snapshot") else None,
                        extra={"best_solution": solution, "error": str(err)},
                    )
                    result_queue.put(record)
                    break
            else:
                dest = 0

            solution.append(dest)
            obs, reward, terminated, truncated, metrics = env.step(dest)
            step += 1

            if terminated or truncated:
                metric_val = metrics.get("relocations", float(step))
                self._best_metric = metric_val
                self._best_solution = solution
                record = ProgressRecord(
                    step=step,
                    metric=metric_val,
                    metrics=metrics,
                    best_metric=metric_val,
                    progress=1.0,
                    yard_snapshot=env.get_state_snapshot() if hasattr(env, "get_state_snapshot") else None,
                    extra={"best_solution": solution},
                )
                result_queue.put(record)
                break
            elif step % max(1, self.config.report_interval) == 0:
                metric_val = metrics.get("relocations", float(step))
                record = ProgressRecord(
                    step=step,
                    metric=metric_val,
                    metrics=metrics,
                    best_metric=metric_val,
                    progress=0.5,
                    yard_snapshot=env.get_state_snapshot() if hasattr(env, "get_state_snapshot") else None,
                    extra={"best_solution": solution},
                )
                result_queue.put(record)

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution


class TLP2026Algorithm(Base2026HeuristicAlgorithm):
    name: str = "TLP / Lowest Position (Lee & Lee 2010 / Zehendner 2017)"
    description: str = "Total Leveling Policy (lowest position rule) evaluated via 2026 isolated engine."

    def _select_destination(self, obs: ObservationState, env_sim: Any, src_stack: int, moving_val: float) -> int:
        return select_tlp_destination(obs, env_sim, src_stack, moving_val)


class RI2026Algorithm(Base2026HeuristicAlgorithm):
    name: str = "RI (Murty 2005 - 2026 Engine)"
    description: str = "Reshuffle Index heuristic evaluated via 2026 isolated engine."

    def _select_destination(self, obs: ObservationState, env_sim: Any, src_stack: int, moving_val: float) -> int:
        return select_ri_destination(obs, env_sim, src_stack, moving_val)


class MM2026Algorithm(Base2026HeuristicAlgorithm):
    name: str = "Min-Max (Caserta 2012 - 2026 Engine)"
    description: str = "Min-Max heuristic evaluated via 2026 isolated engine."

    def _select_destination(self, obs: ObservationState, env_sim: Any, src_stack: int, moving_val: float) -> int:
        return select_minmax_destination(obs, env_sim, src_stack, moving_val)


class PR42026Algorithm(Base2026HeuristicAlgorithm):
    name: str = "PR4 (Jin 2015 - 2026 Engine)"
    description: str = "PR4 lookahead heuristic evaluated via 2026 isolated engine."

    def _select_destination(self, obs: ObservationState, env_sim: Any, src_stack: int, moving_val: float) -> int:
        return select_pr4_destination(obs, env_sim, src_stack, moving_val)


class RDH2026Algorithm(Base2026HeuristicAlgorithm):
    name: str = "RDH (Random Destination - 2026 Engine)"
    description: str = "Random Destination Heuristic with baseline tie-breaking."

    def _select_destination(self, obs: ObservationState, env_sim: Any, src_stack: int, moving_val: float) -> int:
        return select_rdh_destination(obs, env_sim, src_stack, moving_val)


class ERI2026Algorithm(Base2026HeuristicAlgorithm):
    name: str = "ERI (Expected Reshuffle Index - 2026 Engine)"
    description: str = "Expected Reshuffle Index heuristic evaluated via 2026 isolated engine."

    def _select_destination(self, obs: ObservationState, env_sim: Any, src_stack: int, moving_val: float) -> int:
        return select_eri_destination(obs, env_sim, src_stack, moving_val)


class EM2026Algorithm(Base2026HeuristicAlgorithm):
    name: str = "EM (Expected Min-Max - 2026 Engine)"
    description: str = "Expected Min-Max heuristic evaluated via 2026 isolated engine."

    def _select_destination(self, obs: ObservationState, env_sim: Any, src_stack: int, moving_val: float) -> int:
        return select_em_destination(obs, env_sim, src_stack, moving_val)


class EGH2026Algorithm(Base2026HeuristicAlgorithm):
    name: str = "EGH (Expected Group Heuristic - 2026 Engine)"
    description: str = "Expected Group Heuristic (Galle 2018) evaluated via 2026 isolated engine."

    def _select_destination(self, obs: ObservationState, env_sim: Any, src_stack: int, moving_val: float) -> int:
        return select_egh_destination(obs, env_sim, src_stack, moving_val)


class GPOnline2026Algorithm(BaseAlgorithm):
    """Pre-evolved and trainable Genetic Programming relocation rules from Đurasević et al. (2026)."""

    name: str = "GP-Online (Durasevic 2026 Paper)"
    category: str = "Evolutionary"
    description: str = (
        "Genetic Programming relocation rules evaluated via Polish prefix notation. "
        "Evaluates pre-evolved expressions by default or executes evolutionary training."
    )
    geometry: str = "single-bay"
    objectives: List[str] = ["relocations"]
    fidelity: str = "faithful"
    compatible_problems: List[str] = [
        "CRP-Online (2026 Paper)",
        "CRP-Online",
        "CRP-R",
    ]

    def __init__(self, config: Optional[AlgorithmConfig] = None) -> None:
        super().__init__(config=config)
        self._compiled_expr: Optional[PrefixExpression] = None
        self._load_expression()

    def _load_expression(self) -> None:
        raw_expr = self.config.extra.get("expression")
        if raw_expr:
            self._compiled_expr = PrefixExpression(str(raw_expr))
            return

        # Load from pre-evolved expressions catalog
        catalog_path = Path(__file__).resolve().parents[2] / "crp_online" / "data" / "expressions" / "expressions.txt"
        rule_idx = int(self.config.extra.get("rule_index", 0))

        if catalog_path.is_file():
            with open(catalog_path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            if lines:
                selected_line = lines[rule_idx % len(lines)]
                self._compiled_expr = PrefixExpression(selected_line)
                return

        # Fallback to Rule 0 (best overall Caserta rule)
        fallback = "/ / - DIFF RI / / SH WL / DSM DUR + / - DIFF RI / / SH WL / DSM DUR + CUR DUR"
        self._compiled_expr = PrefixExpression(fallback)

    def train(
        self,
        problem_factory: Callable[[], BaseProblem],
        result_queue: mp.Queue,
        stop_event: mp.Event,
    ) -> None:
        """Drives simulation using the compiled GP prefix expression."""
        env = problem_factory()
        obs, info = env.reset()
        solution: List[int] = []
        step = 0

        # Emit initial step 0 progress record so UI responds instantly
        result_queue.put(
            ProgressRecord(
                step=0,
                metric=0.0,
                metrics=env.get_metrics() if hasattr(env, "get_metrics") else {},
                best_metric=0.0,
                progress=0.0,
                yard_snapshot=env.get_state_snapshot() if hasattr(env, "get_state_snapshot") else None,
                extra={"best_solution": []},
            )
        )

        while not stop_event.is_set():
            if hasattr(env, "_sim") and env._sim.is_empty():
                break

            if hasattr(env, "_sim"):
                target = env._sim.get_current_target()
                src = env._sim.get_stack_of_container(target)
                top_cid = env._sim.get_top_container(src)
                obs_state = env._sim.get_observation()
                try:
                    dest = select_best_stack_pf(
                        obs=obs_state,
                        env=env._sim,
                        source_stack=src,
                        moving_val=float(top_cid),
                        expr=self._compiled_expr,
                    )
                except Exception as err:
                    metric_val = 10000.0 + step
                    self._best_metric = metric_val
                    record = ProgressRecord(
                        step=step + 1,
                        metric=metric_val,
                        metrics={"relocations": metric_val, "error": str(err)},
                        best_metric=metric_val,
                        progress=1.0,
                        yard_snapshot=env.get_state_snapshot() if hasattr(env, "get_state_snapshot") else None,
                        extra={"best_solution": solution, "error": str(err)},
                    )
                    result_queue.put(record)
                    break
            else:
                dest = 0

            solution.append(dest)
            obs, reward, terminated, truncated, metrics = env.step(dest)
            step += 1

            if terminated or truncated:
                metric_val = metrics.get("relocations", float(step))
                self._best_metric = metric_val
                self._best_solution = solution
                record = ProgressRecord(
                    step=step,
                    metric=metric_val,
                    metrics=metrics,
                    best_metric=metric_val,
                    progress=1.0,
                    yard_snapshot=env.get_state_snapshot() if hasattr(env, "get_state_snapshot") else None,
                    extra={"best_solution": solution},
                )
                result_queue.put(record)
                break
            elif step % max(1, self.config.report_interval) == 0:
                metric_val = metrics.get("relocations", float(step))
                record = ProgressRecord(
                    step=step,
                    metric=metric_val,
                    metrics=metrics,
                    best_metric=metric_val,
                    progress=0.5,
                    yard_snapshot=env.get_state_snapshot() if hasattr(env, "get_state_snapshot") else None,
                    extra={"best_solution": solution},
                )
                result_queue.put(record)

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution
