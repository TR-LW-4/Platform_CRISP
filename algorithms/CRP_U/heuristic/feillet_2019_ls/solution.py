"""
Internal solution representation for Feillet, Parragh, Tricoire (2019) LS.

A BRP solution is a flat ordered sequence of Steps covering all N containers:
  - relocations  (dst > 0): move container from src to dst
  - retrievals   (dst = 0): remove container from bay

The greedy initialiser embedded here (MinMax rBRP-style) makes this module
**fully self-contained** — no coupling to GLAH or any other algorithm.

Coupling decision
-----------------
The LS operator OPT(n) only requires
  (a) a List[Step] representing the current solution
  (b) init_stacks: the initial bay configuration as a 1-indexed list of
      priority lists (bottom-to-top)

No GLAH data structures (GlahLayout, GlahOp, GlahState) are used.
Any constructive heuristic that can produce a List[Step] (or that can be
converted to one) can therefore serve as the warm-start.  GLAH output can
be accepted via ops_to_steps() if desired, but is not required.

Reference
---------
D. Feillet, S. Parragh, F. Tricoire,
"A local-search based heuristic for the unrestricted block relocation
 problem",
Computers & Operations Research 108 (2019) 44–56.
https://doi.org/10.1016/j.cor.2019.04.005
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════ #
#  Step dataclass                                                     #
# ═══════════════════════════════════════════════════════════════════ #

@dataclass(frozen=True)
class Step:
    """
    One crane operation in the BRP solution.

    container : priority of the moved container (1..N; 1 = retrieved first)
    src       : source stack, 1-indexed
    dst       : destination stack, 1-indexed;  0 = retrieval (container
                leaves the bay)
    """
    container: int
    src:       int
    dst:       int


# ═══════════════════════════════════════════════════════════════════ #
#  Greedy initial solution  (fully self-contained)                   #
# ═══════════════════════════════════════════════════════════════════ #

def build_greedy(
    init_stacks: List[List[int]],
    W:           int,
    H_max:       int,
    N:           int,
) -> List[Step]:
    """
    MinMax greedy for CRP-U: initial solution without any external dependency.

    Policy
    ------
    Retrieve containers in priority order 1 .. N.  For each target,
    relocate only the topmost blocker above it, choosing the destination
    stack that minimises dif(c, d_dst):

        dif(c, d) = d − c        if d > c   (c will be well-located, range 1..N)
                    2N + 1 − d   otherwise  (c will be badly placed, range N+2..2N+1)

    Choosing the smallest dif minimises the chance of future re-relocation.
    This is a conservative rBRP policy (no proactive moves).  The LS operator
    is insensitive to initial solution quality, so a cheap greedy suffices.

    Parameters
    ----------
    init_stacks : 1-indexed list; init_stacks[s] = bottom-to-top priority list
                  for stack s=1..W.  Index 0 is unused (set to None by caller).
    W           : number of stacks
    H_max       : maximum stack height
    N           : number of containers (priorities 1..N)

    Returns
    -------
    Ordered list of Steps (relocations + retrievals) representing a complete
    solution.
    """
    stacks: List[Optional[List[int]]] = [None] + [list(init_stacks[s]) for s in range(1, W + 1)]
    steps:  List[Step] = []

    loc: Dict[int, int] = {}
    for s in range(1, W + 1):
        for c in stacks[s]:  # type: ignore[union-attr]
            loc[c] = s

    for target in range(1, N + 1):
        src = loc.get(target)
        if src is None:
            continue

        while stacks[src] and stacks[src][-1] != target:  # type: ignore[index]
            c = stacks[src][-1]  # type: ignore[index]

            best_dst:   Optional[int] = None
            best_score: int = 10 ** 9
            for dst in range(1, W + 1):
                if dst == src or len(stacks[dst]) >= H_max:  # type: ignore[arg-type]
                    continue
                d = min(stacks[dst]) if stacks[dst] else N + 1  # type: ignore[arg-type]
                score = (d - c) if d > c else (2 * N + 1 - d)
                if score < best_score:
                    best_score = score
                    best_dst   = dst

            if best_dst is None:
                break  # no room; degenerate instance

            steps.append(Step(c, src, best_dst))
            stacks[src].pop()  # type: ignore[union-attr]
            stacks[best_dst].append(c)  # type: ignore[union-attr]
            loc[c] = best_dst

        if stacks[src] and stacks[src][-1] == target:  # type: ignore[index]
            steps.append(Step(target, src, 0))
            stacks[src].pop()  # type: ignore[union-attr]
            del loc[target]

    return steps


# ═══════════════════════════════════════════════════════════════════ #
#  Solution update after OPT(n)                                      #
# ═══════════════════════════════════════════════════════════════════ #

def rebuild_solution(
    steps:    List[Step],
    n:        int,
    events:   List[Tuple[int, int, int]],
    s_final:  int,
    steps_mn: List[Step],
) -> List[Step]:
    """
    Replace container n's moves in *steps* with a new relocation sequence.

    Parameters
    ----------
    steps    : current full solution
    n        : container being reoptimised
    events   : list of (t_before, src, dst) in forward order.
               "Before S^-n step t_before (1-indexed), move n from src to dst."
    s_final  : stack where n resides just before its retrieval in the new sol.
    steps_mn : S^-n steps (steps in `steps` before n's retrieval that are not
               n's own relocations; pre-computed by _build_s_minus_n).

    Returns
    -------
    New flat step list with n's moves replaced.
    """
    n_ret_idx = next(
        (i for i, s in enumerate(steps) if s.container == n and s.dst == 0),
        None,
    )
    steps_after = steps[n_ret_idx + 1:] if n_ret_idx is not None else []

    insert: Dict[int, Tuple[int, int]] = {t: (src, dst) for t, src, dst in events}

    new_steps: List[Step] = []
    for k, smn_step in enumerate(steps_mn, 1):
        if k in insert:
            src, dst = insert[k]
            new_steps.append(Step(n, src, dst))
        new_steps.append(smn_step)

    new_steps.append(Step(n, s_final, 0))
    new_steps.extend(steps_after)
    return new_steps
