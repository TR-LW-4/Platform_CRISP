"""
Ascending-T solve driver for LeeHsu2007NetworkFlowIP.

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
    clone_stacks,
    coarsen_types,
    count_bad_overlaps,
    distinct_types,
    greedy_upper_bound_moves,
    is_fully_well_located,
)
from .extensions import add_exact_final_layout, add_one_type_per_stack
from .mip_model import ModelDims, build_basic_model
from .ordering import OrderingFailure, reconstruct_move_sequence


@dataclass
class SolveResult:
    moves: List[Move] = field(default_factory=list)
    solved: bool = False
    proved_optimal: bool = False
    objective: Optional[float] = None
    T_used: Optional[int] = None
    n_types: int = 0
    n_types_coarsened_from: Optional[int] = None
    n_binaries: int = 0
    gurobi_status: Optional[int] = None
    remaining_bad_overlaps: int = 0
    elapsed_s: float = 0.0
    attempts: int = 0
    skipped_reason: Optional[str] = None
    ordering_failed: bool = False


def _estimate_binaries(n_stacks: int, max_tiers: int, n_types: int, T: int) -> int:
    S, H, C = n_stacks, max_tiers, n_types
    n_cu = max(T - 1, 0) * S * max(H - 1, 0) * C
    n_cd = max(T - 1, 0) * S * max(H - 1, 0) * C
    n_ci = T * S * H * C
    n_co = max(T - 1, 0) * S * H * C
    n_cr = max(T - 1, 0) * S * max(S - 1, 0) * C
    return n_cu + n_cd + n_ci + n_co + n_cr


def _t_candidates(t_start: int, t_max: int, t_step: int) -> List[int]:
    out = []
    t = t_start
    while t <= t_max:
        out.append(t)
        t += max(1, t_step)
    if not out:
        out = [t_start]
    return out


def solve_lee_hsu(
    stacks_init: Stacks,
    max_tiers: int,
    t_start: int = 4,
    t_max: int = 12,
    t_step: int = 2,
    auto_size_t: bool = True,
    max_moves_per_segment: int = 3,
    cycle_breaking_level: int = 3,
    per_solve_time_limit_s: float = 30.0,
    overall_time_budget_s: float = 120.0,
    max_types: int = 12,
    max_binaries: int = 250_000,
    exact_final_layout_demand: Optional[Dict[Tuple[int, int], Optional[int]]] = None,
    one_type_per_stack: bool = False,
) -> SolveResult:
    t0 = time.perf_counter()

    if is_fully_well_located(stacks_init):
        return SolveResult(
            moves=[], solved=True, proved_optimal=True, objective=0.0,
            T_used=0, n_types=len(distinct_types(stacks_init)),
            elapsed_s=time.perf_counter() - t0,
        )

    coarse_stacks, mapping = coarsen_types(stacks_init, max_types)
    n_types_full = len(distinct_types(stacks_init))
    n_types_used = len(set(mapping.values()))
    coarsened_from = n_types_full if n_types_used < n_types_full else None

    n_stacks = len(coarse_stacks)
    candidates = _t_candidates(t_start, t_max, t_step)
    if auto_size_t:
        ub_moves = greedy_upper_bound_moves(coarse_stacks, max_tiers)
        # With multi-move relaxation several moves can share a segment;
        # a modest number of segments usually suffices in practice.
        seed_t = max(t_start, min(t_max, max(3, (ub_moves // max(1, max_moves_per_segment)) + 2)))
        if seed_t not in candidates:
            candidates = sorted(set(candidates + [seed_t]))

    result = SolveResult(n_types=n_types_full, n_types_coarsened_from=coarsened_from)
    result.attempts = 0

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        result.skipped_reason = f"gurobi_unavailable: {e}"
        result.elapsed_s = time.perf_counter() - t0
        return result

    for T in candidates:
        elapsed_so_far = time.perf_counter() - t0
        remaining_budget = overall_time_budget_s - elapsed_so_far
        if remaining_budget <= 0.5:
            result.skipped_reason = result.skipped_reason or "overall_time_budget_exhausted"
            break

        n_binaries = _estimate_binaries(n_stacks, max_tiers, n_types_used, T)
        if n_binaries > max_binaries:
            result.skipped_reason = (
                f"model_too_large T={T} binaries~={n_binaries} > max_binaries={max_binaries}"
            )
            continue

        result.attempts += 1
        dims = ModelDims(
            n_stacks=n_stacks,
            max_tiers=max_tiers,
            type_values=list(range(1, n_types_used + 1)),
            n_time_points=T,
        )
        # coarse_stacks already uses 1..n_types_used bucket values.
        time_limit = max(1.0, min(per_solve_time_limit_s, remaining_budget))

        m, V, stack_order = build_basic_model(
            stacks_init=coarse_stacks,
            dims=dims,
            time_limit_s=time_limit,
            max_moves_per_segment=max_moves_per_segment,
            cycle_breaking_level=cycle_breaking_level,
        )

        if exact_final_layout_demand is not None:
            mapped_demand = {
                (s, h): (mapping.get(v) if v is not None else None)
                for (s, h), v in exact_final_layout_demand.items()
            }
            add_exact_final_layout(m, V, dims, mapped_demand)
        if one_type_per_stack:
            add_one_type_per_stack(m, V, dims)

        m.optimize()
        result.n_binaries = n_binaries
        result.gurobi_status = int(m.Status)

        if m.SolCount <= 0:
            continue

        segment_moves: Dict[int, List[Move]] = {t: [] for t in range(T - 1)}
        for (t, s, z, c), var in V.cr.items():
            if var.X > 0.5:
                segment_moves[t].append((stack_order[s], stack_order[z]))

        try:
            moves, final_coarse = reconstruct_move_sequence(stacks_init, segment_moves, max_tiers)
        except OrderingFailure:
            result.ordering_failed = True
            continue

        final_stacks = apply_moves(stacks_init, moves, max_tiers)
        bad = count_bad_overlaps(final_stacks) if exact_final_layout_demand is None else 0
        if exact_final_layout_demand is not None:
            ok = all(
                (final_stacks.get(s, [None] * (h + 1))[h] if h < len(final_stacks.get(s, [])) else None) == v
                for (s, h), v in exact_final_layout_demand.items()
            )
            bad = 0 if ok else 1

        result.moves = moves
        result.solved = bad == 0
        result.remaining_bad_overlaps = bad
        result.objective = float(m.ObjVal) if m.SolCount > 0 else None
        result.proved_optimal = m.Status == GRB.OPTIMAL
        result.T_used = T

        if result.solved:
            break

    result.elapsed_s = time.perf_counter() - t0
    return result
