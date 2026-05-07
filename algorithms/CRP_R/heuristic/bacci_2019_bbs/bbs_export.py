"""
Convert the platform's Yard / ProblemConfig to the BC-RBRP input format
expected by the BBS heuristic binary.

BC-RBRP input format (see BC_RBRP.cpp / rBRP_BSheu.cpp)
---------------------------------------------------------
Line 1:  w h n
  w = total number of stacks (num_bays × num_rows)
  h = maximum stack height (max_tiers)
  n = total number of containers
Remaining w lines, one per stack (left-to-right, bay-major order):
  k item1 item2 ...
  k   = current height of this stack (number of containers)
  items are priorities 1 … n, listed bottom-to-top
  (priority 1 = retrieved first; matches the platform's convention)

Empty stacks are written as a single "0" on their line.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from core.base_problem import ProblemConfig


def yard_to_bbs_instance(cfg: "ProblemConfig", yard) -> str:
    """
    Convert the current yard state to a BC-RBRP instance string.

    Parameters
    ----------
    cfg  : ProblemConfig with num_bays, num_rows, max_tiers, num_containers
    yard : core.yard.Yard instance

    Returns
    -------
    Multi-line string ready to be written to a temp file.
    """
    nb = int(cfg.num_bays)
    nr = int(cfg.num_rows)
    w  = nb * nr
    h  = int(cfg.max_tiers)
    n  = int(cfg.num_containers)

    lines: List[str] = [f"{w} {h} {n}"]

    placed = 0
    for idx in range(w):
        bay = idx // nr + 1
        row = idx % nr + 1
        stk = yard.stacks.get((bay, row))
        if stk is None or stk.is_empty:
            lines.append("0")
        else:
            items = [int(c.priority) for c in stk.containers]   # bottom → top
            placed += len(items)
            lines.append(f"{len(items)} " + " ".join(str(p) for p in items))

    if placed != n:
        raise ValueError(
            f"BBS export: yard holds {placed} containers but "
            f"ProblemConfig.num_containers={n}."
        )

    return "\n".join(lines) + "\n"
