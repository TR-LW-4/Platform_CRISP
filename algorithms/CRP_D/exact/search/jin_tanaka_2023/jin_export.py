"""
Yard → Jin–Tanaka IDBB binary input for JinTanaka2023ForCRPD.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from core.base_problem import ProblemConfig
    from core.yard import Yard


def yard_to_jin_instance(cfg: "ProblemConfig", yard: "Yard") -> str:
    """Return the full instance text expected by Jin's ``main-solve``."""
    nb      = int(cfg.num_bays)
    nr      = int(cfg.num_rows)
    n_stacks = nb * nr
    n_blocks = int(cfg.num_containers)
    n_tiers  = int(cfg.max_tiers)

    lines: List[str] = [f"{n_stacks} {n_tiers} {n_blocks}"]

    placed = 0
    for idx in range(n_stacks):
        bay = idx // nr + 1
        row = idx  % nr + 1
        stk = yard.stacks[(bay, row)]
        h   = len(stk.containers)
        placed += h
        prios = " ".join(str(int(c.priority)) for c in stk.containers)
        lines.append(f"{h} {prios}" if h > 0 else "0")

    if placed != n_blocks:
        raise ValueError(
            f"Jin/CRP-D export: yard holds {placed} containers but "
            f"ProblemConfig.num_containers={n_blocks}."
        )

    return "\n".join(lines) + "\n"
