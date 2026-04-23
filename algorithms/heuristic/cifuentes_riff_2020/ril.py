"""
Reshuffle-Index with Look-ahead (RIL) repair heuristic.

Reference
---------
K.-C. Wu and C.-J. Ting, "A Beam Search Algorithm for Minimizing
Reshuffle Operations at Container Yards", Proceedings of the
International Conference on Logistics and Maritime Systems (2010).

Role in G-CREM
--------------
The Cifuentes & Riff (2020) GRASP (``G-CREM``) uses RIL as the
"repair operator" inside its local-search / hill-climbing phase.
When one relocation of the incumbent plan is edited, the remainder of
the plan is re-built by deterministically picking destination stacks
through RIL.

Rule
----
Given a topmost blocker ``c`` at stack ``src`` that needs to be relocated,
for every candidate destination stack ``d`` (non-source, non-full):

    RI(d)      = number of containers currently in d whose priority is
                 strictly smaller than c.priority.  These are containers
                 that MUST be retrieved before c, which would force c to
                 be re-relocated later.
    min_pri(d) = lowest priority currently residing in d (``+inf`` if d
                 is empty), i.e. the "most-urgent" container in d.  Used
                 as look-ahead tie-breaker: smaller = the stack will be
                 "opened" sooner, so our blocker becomes accessible again
                 sooner as well.

Pick the stack minimising ``(RI(d), min_pri(d))`` lexicographically.
Empty stacks are preferred (RI = 0, min_pri = +inf makes them always
best under ties with other RI-0 stacks when no other RI-0 stack has a
finite min_pri; they still beat RI > 0 stacks).  Ties on both keys are
broken deterministically by (bay, row).
"""

from __future__ import annotations

from typing import Optional, Tuple


_INF = float("inf")


def ril_select_dst(
    sim,
    blocker_priority: int,
    src_pos:          Tuple[int, int],
    max_tiers:        int,
) -> Optional[Tuple[int, int]]:
    """
    Pick a destination stack for ``blocker`` using the RIL rule.

    Parameters
    ----------
    sim              : the (simulated) Yard state at the moment of decision
    blocker_priority : priority of the blocker to be relocated
    src_pos          : (bay, row) of the source stack (never returned)
    max_tiers        : yard capacity per stack

    Returns
    -------
    (bay, row) of the chosen destination stack, or ``None`` if every
    non-source stack is full (yard is completely packed).
    """
    best:       Optional[Tuple[int, int]] = None
    best_key:   Tuple[float, float, int, int] = (_INF, _INF, _INF, _INF)

    for (bay, row), stk in sim.stacks.items():
        if (bay, row) == src_pos:
            continue
        if stk.height >= max_tiers:
            continue

        if stk.is_empty:
            ri      = 0
            min_pri = _INF
        else:
            ri      = sum(
                1 for c in stk.containers if c.priority < blocker_priority
            )
            min_pri = float(min(c.priority for c in stk.containers))

        key = (float(ri), min_pri, bay, row)
        if key < best_key:
            best_key = key
            best     = (bay, row)

    return best
