"""
Section 6 iterative T-search driver for IPS6.

Unlike Lee & Hsu (2007) or de Melo da Silva et al. (2018) -- both of which
size their model from a heuristic upper bound of unknown tightness (and,
per this paper's own Section 6, the de Melo da Silva heuristic can loop
forever without returning a value on some instance shapes) -- this paper's
own recommended procedure does not need an upper bound at all:

  1. Start at the smallest T that *cannot* be ruled out: T = LB + 1 time
     points (LB relocation-lower-bound segments), where LB is the cheap
     "blocking containers" count from Section 3.
  2. Solve. If Gurobi finds a feasible solution, it is automatically
     *optimal* (there is no slack to exploit at the smallest feasible T).
  3. Otherwise increase T by one time point (one more segment) and repeat.

This also means the model does not need the "push moves as early as
possible" constraint (Eq. 29): the paper notes it is only useful when
solving with a single, deliberately oversized T.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .core import (
    Move,
    Stacks,
    apply_moves,
    blocking_containers_lower_bound,
    coarsen_priorities,
    count_bad_overlaps,
    distinct_priorities,
    is_fully_well_located,
)
from .extensions import (
    add_balance_constraint,
    add_no_empty_or_full_stacks,
    add_same_priority_bonus,
    add_stability_bonus,
)
from .mip_model import ModelDims, build_ips6_model


@dataclass
class SolveResult:
    moves: List[Move] = field(default_factory=list)
    solved: bool = False
    proved_optimal: bool = False
    objective: Optional[float] = None
    lower_bound: Optional[int] = None
    n_time_points_used: Optional[int] = None
    n_priorities: int = 0
    n_priorities_coarsened_from: Optional[int] = None
    n_binaries: int = 0
    gurobi_status: Optional[int] = None
    remaining_bad_overlaps: int = 0
    elapsed_s: float = 0.0
    attempts: int = 0
    skipped_reason: Optional[str] = None


def _estimate_binaries(n_stacks: int, max_tiers: int, n_priorities: int, n_time_points: int) -> int:
    S, H, P, TP = n_stacks, max_tiers, n_priorities, n_time_points
    n_segments = max(TP - 1, 0)
    return TP * S * H * P + 2 * n_segments * S * H * P


def solve_ips6(
    stacks_init: Stacks,
    max_tiers: int,
    transitive_move_breaking: bool = True,
    fix_last_segment_vars: bool = True,
    continuous_move_vars: bool = False,
    balance_kappa: Optional[int] = None,
    no_empty_or_full_stacks: bool = False,
    stability_bonus: bool = False,
    same_priority_bonus: bool = False,
    per_solve_time_limit_s: float = 30.0,
    overall_time_budget_s: float = 120.0,
    max_priorities: int = 20,
    max_binaries: int = 250_000,
    max_t_attempts: int = 40,
) -> SolveResult:
    t0 = time.perf_counter()

    if is_fully_well_located(stacks_init):
        return SolveResult(
            moves=[], solved=True, proved_optimal=True, objective=0.0,
            n_time_points_used=1, lower_bound=0,
            n_priorities=len(distinct_priorities(stacks_init)),
            elapsed_s=time.perf_counter() - t0,
        )

    coarse_stacks, mapping = coarsen_priorities(stacks_init, max_priorities)
    n_priorities_full = len(distinct_priorities(stacks_init))
    n_priorities_used = len(set(mapping.values()))
    coarsened_from = n_priorities_full if n_priorities_used < n_priorities_full else None

    n_stacks = len(coarse_stacks)
    lb = blocking_containers_lower_bound(coarse_stacks)
    lb = max(lb, 1)

    result = SolveResult(
        n_priorities=n_priorities_full,
        n_priorities_coarsened_from=coarsened_from,
        lower_bound=lb,
    )

    try:
        import gurobipy as gp  # noqa: F401
        from gurobipy import GRB
    except Exception as e:
        result.skipped_reason = f"gurobi_unavailable: {e}"
        result.elapsed_s = time.perf_counter() - t0
        return result

    total_containers = sum(len(v) for v in coarse_stacks.values())

    for n_segments in range(lb, lb + max_t_attempts):
        n_time_points = n_segments + 1

        elapsed_so_far = time.perf_counter() - t0
        remaining_budget = overall_time_budget_s - elapsed_so_far
        if remaining_budget <= 0.5:
            result.skipped_reason = result.skipped_reason or "overall_time_budget_exhausted"
            break

        n_binaries = _estimate_binaries(n_stacks, max_tiers, n_priorities_used, n_time_points)
        if n_binaries > max_binaries:
            result.skipped_reason = (
                f"model_too_large n_time_points={n_time_points} binaries~={n_binaries} > max_binaries={max_binaries}"
            )
            continue

        result.attempts += 1
        dims = ModelDims(
            n_stacks=n_stacks,
            max_tiers=max_tiers,
            priority_values=list(range(1, n_priorities_used + 1)),
            n_time_points=n_time_points,
        )
        time_limit = max(1.0, min(per_solve_time_limit_s, remaining_budget))

        m, V, stack_order = build_ips6_model(
            stacks_init=coarse_stacks,
            dims=dims,
            time_limit_s=time_limit,
            earliest_time_push=False,
            transitive_move_breaking=transitive_move_breaking,
            fix_last_segment_vars=fix_last_segment_vars,
            continuous_move_vars=continuous_move_vars,
        )

        if balance_kappa is not None:
            add_balance_constraint(m, V, dims, balance_kappa)
        if no_empty_or_full_stacks:
            add_no_empty_or_full_stacks(m, V, dims)
        if stability_bonus:
            add_stability_bonus(m, V, dims)
        if same_priority_bonus:
            add_same_priority_bonus(m, V, dims, total_containers)

        m.optimize()
        result.n_binaries = n_binaries
        result.gurobi_status = int(m.Status)

        if m.SolCount <= 0:
            continue

        moves: List[Move] = []
        for t in range(n_segments):
            src_stack = None
            dst_stack = None
            for si in range(n_stacks):
                for h in range(max_tiers):
                    for p in range(dims.n_priorities):
                        if dst_stack is None and V.z[(t, si, h, p)].X > 0.5:
                            dst_stack = stack_order[si]
                        if src_stack is None and V.w[(t, si, h, p)].X > 0.5:
                            src_stack = stack_order[si]
            if src_stack is not None and dst_stack is not None:
                moves.append((src_stack, dst_stack))

        final_stacks = apply_moves(stacks_init, moves, max_tiers)
        bad = count_bad_overlaps(final_stacks)
        total_before = sum(len(v) for v in stacks_init.values())
        total_after = sum(len(v) for v in final_stacks.values())
        solved = bad == 0 and total_before == total_after

        result.moves = moves
        result.solved = solved
        result.remaining_bad_overlaps = bad
        result.objective = float(m.ObjVal) if m.SolCount > 0 else None
        result.proved_optimal = m.Status == GRB.OPTIMAL
        result.n_time_points_used = n_time_points

        # The first feasible T in the ascending search is, by construction
        # of the iterative procedure, already optimal -- for the (possibly
        # priority-coarsened) instance actually given to the solver. Always
        # stop here: escalating T further cannot un-collapse a coarsening
        # bucket, so if `solved` is still False purely because of residual
        # bad overlaps between merged priorities, no larger T will fix it
        # either, and continuing would just burn the whole time budget.
        break

    result.elapsed_s = time.perf_counter() - t0
    return result
