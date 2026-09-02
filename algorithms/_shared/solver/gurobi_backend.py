"""
Shared Gurobi solve/export helpers for MIP algorithms.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Optional, Tuple


def build_model(
    time_limit_s: float = 3600.0,
    mip_gap_abs: float = 0.99,
    output_flag: int = 0,
    threads: int = 0,
):
    """
    Create a Gurobi Model with standard platform parameters.

    Parameters
    ----------
    time_limit_s : wall-clock time limit in seconds.
    mip_gap_abs  : absolute MIP gap (0.99 → stop at gap < 1, i.e., integer
                   optimality because objective is always an integer).
    output_flag  : 0 = silent, 1 = verbose Gurobi output.
    threads      : 0 = use all available cores.
    """
    import gurobipy as gp

    m = gp.Model()
    m.Params.TimeLimit  = float(time_limit_s)
    m.Params.MIPGapAbs  = float(mip_gap_abs)
    m.Params.OutputFlag = int(output_flag)
    if threads > 0:
        m.Params.Threads = int(threads)
    return m


def parse_result(model) -> Tuple[Optional[int], bool]:
    """
    Interpret Gurobi solve status → (objective_value, optimal_proven).

    Returns (None, False) when no feasible solution was found.
    """
    from gurobipy import GRB

    status = model.Status
    if model.SolCount == 0:
        return None, False
    obj = int(round(model.ObjVal))
    optimal = status == GRB.OPTIMAL
    return obj, optimal


def gurobi_available() -> bool:
    """Return True iff gurobipy is importable and a valid license exists."""
    try:
        import gurobipy as gp
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
        return True
    except Exception:
        return False
