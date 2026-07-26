"""
Solver-assisted path-reduction core (Gurobi).

This module implements a binary IP selection over merge candidates:
each candidate removes one movement from the current feasible sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Set, Tuple

Move = Tuple[int, int]
ExecutedMove = Tuple[int, int, int]  # (priority, src, dst)


@dataclass(frozen=True)
class MergeCandidate:
    first_idx: int
    second_idx: int
    new_dst: int
    gain: int = 1


def extract_merge_candidates(executed: Sequence[ExecutedMove]) -> List[MergeCandidate]:
    cands: List[MergeCandidate] = []
    n = len(executed)
    for i in range(n):
        p_i, a, b = executed[i]
        for j in range(i + 1, n):
            p_j, b2, c = executed[j]
            if p_i != p_j or b2 != b:
                continue
            # Same simple feasibility guard as minor sequence rule.
            blocked = any(executed[k][2] == c for k in range(i + 1, j))
            if blocked:
                continue
            cands.append(MergeCandidate(first_idx=i, second_idx=j, new_dst=c, gain=1))
    return cands


def solve_best_merges(
    n_moves: int,
    candidates: Sequence[MergeCandidate],
    time_limit: float = 5.0,
) -> Set[int]:
    """
    Select non-conflicting merge candidates maximizing removed moves.
    Returns set of selected candidate indices.
    """
    if not candidates:
        return set()

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        raise RuntimeError(
            "Gurobi is required by Lee–Chao (2009) NS+IP solver subroutine."
        ) from e

    m = gp.Model("lee_chao_2009_path_reduce")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = max(0.1, float(time_limit))

    x = {k: m.addVar(vtype=GRB.BINARY, name=f"x_{k}") for k in range(len(candidates))}

    # Objective: maximize removed moves.
    m.setObjective(gp.quicksum(candidates[k].gain * x[k] for k in x), GRB.MAXIMIZE)

    # No move index can be touched by more than one selected candidate.
    for mv_idx in range(n_moves):
        touched = [
            k for k, cand in enumerate(candidates)
            if cand.first_idx == mv_idx or cand.second_idx == mv_idx
        ]
        if touched:
            m.addConstr(gp.quicksum(x[k] for k in touched) <= 1)

    m.optimize()

    chosen: Set[int] = set()
    for k in x:
        if x[k].X > 0.5:
            chosen.add(k)
    return chosen


def apply_merges(
    sequence: Sequence[Move],
    candidates: Sequence[MergeCandidate],
    selected: Set[int],
) -> List[Move]:
    out = list(sequence)
    if not selected:
        return out

    # Process by descending second_idx so deletions remain stable.
    ordered = sorted((candidates[k] for k in selected), key=lambda c: c.second_idx, reverse=True)
    for cand in ordered:
        if cand.first_idx >= len(out) or cand.second_idx >= len(out):
            continue
        src, _old_dst = out[cand.first_idx]
        out[cand.first_idx] = (src, cand.new_dst)
        del out[cand.second_idx]
    return out

