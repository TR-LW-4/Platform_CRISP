from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .branch_and_bound import BBSolveResult, solve_cpmpct_branch_and_bound
from .core import Stacks
from .crane_time import CraneParams


@dataclass
class MIPSolveResult:
    moves: List[tuple]
    crane_time: float
    solved: bool
    elapsed_s: float
    note: str


def solve_cpmpct_mip(
    stacks_init: Stacks,
    max_tiers: int,
    p: CraneParams,
    time_limit_s: float,
) -> MIPSolveResult:
    """
    Embedding note:
    The paper presents a full time-indexed IPCT model.
    In this first embedding stage we keep the MIP entrypoint and route to the
    exact B&B engine to provide a stable solver path in-platform.
    """
    res: BBSolveResult = solve_cpmpct_branch_and_bound(
        stacks_init=stacks_init,
        max_tiers=max_tiers,
        p=p,
        time_limit_s=time_limit_s,
    )
    return MIPSolveResult(
        moves=list(res.moves),
        crane_time=float(res.crane_time),
        solved=bool(res.solved),
        elapsed_s=float(res.elapsed_s),
        note="mip_backend_routed_to_bb_v1",
    )
