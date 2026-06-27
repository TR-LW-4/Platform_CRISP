"""
Export a CRP-D yard snapshot to Tanaka ``restricted-duplicate`` input format.

Tanaka's reader (``main.c``) expects:
  Line 1: ``n_stack n_block``
  Line 2: whitespace-separated sequence of ``height priority_0 … priority_{h-1}``
           repeated for each stack 0 … n_stack-1 (bottom → top, tier 0 first).

This format is identical to the CRP-R export; the binary natively handles
duplicate priorities (group IDs).  A group ID repeated across containers is
valid input — the solver branches on target-block selection within the group.

Stack order: index i → bay i // num_rows + 1, row i % num_rows + 1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from core.base_problem import ProblemConfig
    from core.yard import Yard


def yard_to_tanaka_instance(cfg: "ProblemConfig", yard: "Yard") -> str:
    """
    Return full problem text for Tanaka ``brp_bb`` (restricted-duplicate binary).

    Uses ``container.priority`` as the group ID, which the binary treats as a
    block priority.  Duplicate values → duplicate-priority problem instance.
    """
    nb      = int(cfg.num_bays)
    nr      = int(cfg.num_rows)
    n_stack = nb * nr
    exp_n   = int(cfg.num_containers)

    tokens_body: List[str] = []
    placed = 0

    for idx in range(n_stack):
        bay = idx // nr + 1
        row = idx % nr + 1
        stk = yard.stacks[(bay, row)]
        h   = len(stk.containers)
        placed += h
        tokens_body.append(str(h))
        for c in stk.containers:
            tokens_body.append(str(int(c.priority)))

    if placed != exp_n:
        raise ValueError(
            f"Tanaka CRP-D export: yard holds {placed} containers but "
            f"ProblemConfig.num_containers={exp_n}."
        )

    line1 = f"{n_stack} {placed}"
    line2 = " ".join(tokens_body)
    return line1 + "\n" + line2 + "\n"
