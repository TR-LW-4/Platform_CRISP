"""
Time-horizon solve driver for DeMeloSilva2018PMP.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .core import (
    Move,
    Stacks,
    apply_moves,
    best_pmp_heuristic_upper_bound,
    clone_stacks,
    coarsen_types,
    count_bad_overlaps,
    distinct_types,
    is_fully_well_located,
)
from .extensions import add_exact_final_layout, add_sorted_final_layout
from .mip_model import ModelDims, build_pmp_model


@dataclass
class SolveResult:
    moves: List[Move] = field(default_factory=list)
    solved: bool = False
    proved_optimal: bool = False
    objective: Optional[float] = None
    T_used: Optional[int] = None
    heuristic_T: Optional[int] = None
    n_groups: int = 0
    n_groups_coarsened_from: Optional[int] = None
    n_binaries: int = 0
    gurobi_status: Optional[int] = None
    remaining_bad_overlaps: int = 0
    elapsed_s: float = 0.0
    attempts: int = 0
    skipped_reason: Optional[str] = None


def _estimate_binaries(n_stacks: int, max_tiers: int, n_groups: int, R: int) -> int:
    S, H, G = n_stacks, max_tiers, n_groups
    n_x = (R + 1) * G * S * H
    n_y = R * G * S * H
    n_z = R * G * S * H
    return n_x + n_y + n_z


def solve_pmp(
    stacks_init: Stacks,
    max_tiers: int,
    lifo_strengthening: bool = True,
    flow_strengthening: bool = True,
    use_exact_final_layout: bool = False,
    exact_final_layout_demand: Optional[Dict[Tuple[int, int], Optional[int]]] = None,
    per_solve_time_limit_s: float = 30.0,
    overall_time_budget_s: float = 120.0,
    max_types: int = 20,
    max_binaries: int = 250_000,
    t_escalation_step: int = 2,
    t_escalation_max_attempts: int = 5,
) -> SolveResult:
    t0 = time.perf_counter()

    if is_fully_well_located(stacks_init):
        return SolveResult(
            moves=[], solved=True, proved_optimal=True, objective=0.0,
            T_used=0, heuristic_T=0, n_groups=len(distinct_types(stacks_init)),
            elapsed_s=time.perf_counter() - t0,
        )

    coarse_stacks, mapping = coarsen_types(stacks_init, max_types)
    n_groups_full = len(distinct_types(stacks_init))
    n_groups_used = len(set(mapping.values()))
    coarsened_from = n_groups_full if n_groups_used < n_groups_full else None

    n_stacks = len(coarse_stacks)
    heuristic_T, _ = best_pmp_heuristic_upper_bound(coarse_stacks, max_tiers)
    heuristic_T = max(heuristic_T, 1)

    result = SolveResult(
        n_groups=n_groups_full,
        n_groups_coarsened_from=coarsened_from,
        heuristic_T=heuristic_T,
    )

    try:
        import gurobipy as gp  # noqa: F401
        from gurobipy import GRB
    except Exception as e:
        result.skipped_reason = f"gurobi_unavailable: {e}"
        result.elapsed_s = time.perf_counter() - t0
        return result

    mapped_demand = None
    if use_exact_final_layout and exact_final_layout_demand is not None:
        mapped_demand = {
            (s, h): (mapping.get(v) if v is not None else None)
            for (s, h), v in exact_final_layout_demand.items()
        }

    candidates = [heuristic_T + t_escalation_step * i for i in range(max(1, t_escalation_max_attempts))]

    for T in candidates:
        elapsed_so_far = time.perf_counter() - t0
        remaining_budget = overall_time_budget_s - elapsed_so_far
        if remaining_budget <= 0.5:
            result.skipped_reason = result.skipped_reason or "overall_time_budget_exhausted"
            break

        n_binaries = _estimate_binaries(n_stacks, max_tiers, n_groups_used, T)
        if n_binaries > max_binaries:
            result.skipped_reason = (
                f"model_too_large T={T} binaries~={n_binaries} > max_binaries={max_binaries}"
            )
            continue

        result.attempts += 1
        dims = ModelDims(
            n_stacks=n_stacks,
            max_tiers=max_tiers,
            group_values=list(range(1, n_groups_used + 1)),
            n_steps=T,
        )
        time_limit = max(1.0, min(per_solve_time_limit_s, remaining_budget))

        m, V, stack_order = build_pmp_model(
            stacks_init=coarse_stacks,
            dims=dims,
            time_limit_s=time_limit,
            lifo_strengthening=lifo_strengthening,
            flow_strengthening=flow_strengthening,
        )

        if mapped_demand is not None:
            add_exact_final_layout(m, V, dims, mapped_demand)
        else:
            add_sorted_final_layout(m, V, dims)

        m.optimize()
        result.n_binaries = n_binaries
        result.gurobi_status = int(m.Status)

        if m.SolCount <= 0:
            continue

        moves: List[Move] = []
        for t in range(1, T + 1):
            src_stack = None
            dst_stack = None
            for g in range(dims.n_groups):
                for si in range(n_stacks):
                    for h in range(max_tiers):
                        if dst_stack is None and V.y[(t, g, si, h)].X > 0.5:
                            dst_stack = stack_order[si]
                        if src_stack is None and V.z[(t, g, si, h)].X > 0.5:
                            src_stack = stack_order[si]
            if src_stack is not None and dst_stack is not None:
                moves.append((src_stack, dst_stack))

        final_stacks = apply_moves(stacks_init, moves, max_tiers)
        if mapped_demand is not None and exact_final_layout_demand is not None:
            ok = all(
                (final_stacks.get(s, [None] * (h + 1))[h] if h < len(final_stacks.get(s, [])) else None) == v
                for (s, h), v in exact_final_layout_demand.items()
            )
            bad = 0 if ok else 1
        else:
            bad = count_bad_overlaps(final_stacks)

        total_before = sum(len(v) for v in stacks_init.values())
        total_after = sum(len(v) for v in final_stacks.values())
        solved = bad == 0 and total_before == total_after

        result.moves = moves
        result.solved = solved
        result.remaining_bad_overlaps = bad
        result.objective = float(m.ObjVal) if m.SolCount > 0 else None
        result.proved_optimal = m.Status == GRB.OPTIMAL
        result.T_used = T

        if result.solved:
            break

    result.elapsed_s = time.perf_counter() - t0
    return result
