"""
Export a Platform_CRISP yard snapshot to Tanaka ``restricted-duplicate`` input.

Tanaka's reader (``main.c``) expects:
  Line 1: ``n_stack n_block``
  Line 2+: whitespace-separated integers describing stacks in order
           ``index 0 .. n_stack-1``.  For each stack: ``height`` then ``height``
           priorities **bottom → top** (tier 0 first).

Stack order matches CRP-R flat actions: index ``i`` → bay ``i // num_rows + 1``,
row ``i % num_rows + 1``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from core.base_problem import ProblemConfig
    from core.yard import Yard


def yard_to_tanaka_instance(cfg: "ProblemConfig", yard: "Yard") -> str:
    """Return full problem text for Tanaka ``brp_bb`` stdin/file."""
    nb = int(cfg.num_bays)
    nr = int(cfg.num_rows)
    n_stack = nb * nr
    exp_n = int(cfg.num_containers)

    tokens_body: List[str] = []
    placed = 0

    for idx in range(n_stack):
        bay = idx // nr + 1
        row = idx % nr + 1
        stk = yard.stacks[(bay, row)]
        h = len(stk.containers)
        placed += h
        tokens_body.append(str(h))
        for c in stk.containers:
            tokens_body.append(str(int(c.priority)))

    if placed != exp_n:
        raise ValueError(
            f"Tanaka export: yard holds {placed} containers but "
            f"ProblemConfig.num_containers={exp_n}."
        )

    line1 = f"{n_stack} {placed}"
    line2 = " ".join(tokens_body)
    return line1 + "\n" + line2 + "\n"

