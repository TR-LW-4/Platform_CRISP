"""
Zhu, Qin, Lim & Zhang (2012) IDA* search engine — unrestricted variant.

W. Zhu, H. Qin, A. Lim, H. Zhang, "Iterative deepening A* algorithms for
the container relocation problem", IEEE Transactions on Automation
Science and Engineering 9(4) (2012) 710-722.

Self-contained IDA* engine (Algorithm 1 / Algorithm 2 of the paper) for
the *unrestricted* variant (CRP-U): a relocation may move the top
container of *any* stack (not only blockers above the current target)
to any other non-full stack.

Deliberately duplicates the state representation, LB1/LB3 and the
PR1-PR4 destination-selection helpers already written for the
restricted variant (``CRP_R/exact/search/zhu_2012/ida_core.py``) rather
than importing them, so this package stays independent and can be
added/changed/removed without touching CRP_R (same convention as the
platform's other paired R/U implementations, e.g. GLAH).

Components implemented
-----------------------
- Minimum equivalent layout reduction (shared with CRP-R: only the
  *retrieval* semantics are identical between R and U -- only the set
  of legal *relocation* moves differs).
- LB1 (Eq. 1, admissible for both variants) and, optionally, LB3
  (Eqs. 3-4) reused *as-is* from the restricted analysis; the paper
  notes LB3 is **not proven admissible for the unrestricted variant**
  but that the more aggressive pruning it produces can still yield
  better solutions under a strict time budget (IDA*-UM3) -- at the
  cost of never being able to certify optimality.
- Probe heuristics PU1 / PU2 (Section V-B.2), built on top of the
  restricted PR3 / PR4 destination rule with the "relocate a helper
  stack's own local minimum out of the way first" refinement.
- IDA* main loop with an optional transposition table (dict of
  ``state -> best depth seen``) to avoid re-exploring identical
  layouts reached via different move sequences (IDA*-UM).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

State = Tuple[Tuple[int, ...], ...]

PR3, PR4 = "PR3", "PR4"   # base destination rule used by PU1 / PU2
PU1, PU2 = "PU1", "PU2"
LB1, LB3 = "LB1", "LB3"


# ================================================================ #
#  State helpers (identical semantics to the restricted variant)     #
# ================================================================ #

def state_from_stacks(stacks: List[List[int]]) -> State:
    return tuple(tuple(s) for s in stacks)


def _locate(state: State, priority: int) -> Optional[Tuple[int, int]]:
    for si, stack in enumerate(state):
        for hi, p in enumerate(stack):
            if p == priority:
                return si, hi
    return None


def reduce_state(state: State, target: int, n_total: int) -> Tuple[State, int]:
    """Minimum equivalent layout: retrieval semantics are unaffected by
    the unrestricted relocation rule, so this is identical to CRP-R."""
    stacks = list(state)
    while target <= n_total:
        loc = _locate(state, target)
        if loc is None:
            target += 1
            continue
        si, hi = loc
        if hi != len(stacks[si]) - 1:
            break
        stacks[si] = stacks[si][:-1]
        state = tuple(stacks)
        target += 1
    return state, target


def _eligible_dsts(state: State, exclude_idx: int, max_tiers: int) -> List[int]:
    return [i for i, s in enumerate(state) if i != exclude_idx and len(s) < max_tiers]


def _min_priority(stack: Tuple[int, ...]) -> float:
    return min(stack) if stack else float("inf")


def _move_top(state: State, src: int, dst: int) -> State:
    stacks = list(state)
    p = stacks[src][-1]
    stacks[src] = stacks[src][:-1]
    stacks[dst] = stacks[dst] + (p,)
    return tuple(stacks)


# ================================================================ #
#  Lower bounds                                                      #
# ================================================================ #

def _has_safe_dst(state: State, exclude_idx: int, p: int, max_tiers: int) -> bool:
    for i, stack in enumerate(state):
        if i == exclude_idx or len(stack) >= max_tiers:
            continue
        if not stack or min(stack) > p:
            return True
    return False


def lb1(state: State, target: int = 0, n_total: int = 0, max_tiers: int = 0) -> int:
    """LB1 (Eq. 1) -- admissible for both variants."""
    total = 0
    for stack in state:
        min_below: Optional[int] = None
        for p in stack:
            if min_below is not None and p > min_below:
                total += 1
            min_below = p if min_below is None else min(min_below, p)
    return total


def lb3(state: State, target: int, n_total: int, max_tiers: int) -> int:
    """
    LB3 (Eqs. 3-4), reused unchanged from the restricted analysis for
    the more aggressive pruning it offers under IDA*-UM3.  **Not
    proven admissible for the unrestricted variant** (Section V-A) --
    solutions found while using it are never marked optimal.
    """
    total = lb1(state)
    for si, stack in enumerate(state):
        if not stack:
            continue
        t_prime = min(stack)
        h_prime = stack.index(t_prime)
        above = stack[h_prime + 1:]
        if not above:
            continue

        reduced: List[Tuple[int, ...]] = []
        for stk in state:
            cut = len(stk)
            for idx, pp in enumerate(stk):
                if pp < t_prime:
                    cut = idx
                    break
            reduced.append(stk[:cut])

        for p in above:
            if not _has_safe_dst(tuple(reduced), si, p, max_tiers):
                total += 1
    return total


_LB_FUNCS = {LB1: lb1, LB3: lb3}


# ================================================================ #
#  PR3 / PR4 destination rule (used as the base of PU1 / PU2)        #
# ================================================================ #

def _pr3_dst(state: State, src: int, max_tiers: int, p: int) -> Optional[int]:
    elig = _eligible_dsts(state, src, max_tiers)
    if not elig:
        return None
    case_a = [i for i in elig if _min_priority(state[i]) > p]
    if case_a:
        return min(case_a, key=lambda i: _min_priority(state[i]))
    return max(elig, key=lambda i: _min_priority(state[i]))


def _pr4_dst(state: State, src: int, max_tiers: int, p: int) -> Optional[int]:
    elig = _eligible_dsts(state, src, max_tiers)
    if not elig:
        return None
    case_a = [i for i in elig if _min_priority(state[i]) > p]
    if case_a:
        return min(case_a, key=lambda i: _min_priority(state[i]))

    ranked = sorted(elig, key=lambda i: -_min_priority(state[i]))
    best = ranked[0]
    src_sorted = sorted(state[src])
    is_second_smallest = len(src_sorted) >= 2 and p == src_sorted[1]
    only_one_slot = (max_tiers - len(state[best])) == 1
    if (not is_second_smallest) and only_one_slot and len(ranked) > 1:
        return ranked[1]
    return best


def _has_case_a_dst(state: State, src: int, max_tiers: int, p: int) -> bool:
    """True iff a *genuinely* safe destination exists for *p* right now."""
    return any(
        _min_priority(state[i]) > p
        for i in _eligible_dsts(state, src, max_tiers)
    )


_PR_DST_FUNCS = {PR3: _pr3_dst, PR4: _pr4_dst}


# ================================================================ #
#  Probe heuristics PU1 / PU2 (Section V-B.2)                        #
# ================================================================ #

def _find_helper_move(
    state: State, avoid_src: int, max_tiers: int,
) -> Optional[Tuple[int, int]]:
    """
    Look for a stack *j* (other than ``avoid_src``) whose top container
    is *j*'s own local minimum (i.e. "due" to be relocated eventually
    with no cost lost) and which has a genuinely safe destination *k*
    right now.  Among all such *j*, pick the one with the largest top
    value (Section V-B.2).  Returns ``(j, k)`` or ``None``.
    """
    candidates: List[Tuple[int, int, int]] = []  # (top_value, j, k)
    for j, stack in enumerate(state):
        if j == avoid_src or not stack:
            continue
        top = stack[-1]
        if top != min(stack):
            continue
        elig = [
            k for k in _eligible_dsts(state, j, max_tiers)
            if k != avoid_src or True  # dst may legally be avoid_src too
        ]
        safe = [k for k in elig if _min_priority(state[k]) > top]
        if not safe:
            continue
        k = min(safe, key=lambda i: _min_priority(state[i]))
        candidates.append((top, j, k))

    if not candidates:
        return None
    _, j, k = max(candidates, key=lambda t: t[0])
    return j, k


def probe_unrestricted(
    state: State,
    target: int,
    n_total: int,
    max_tiers: int,
    mode: str,
    node_budget: Optional[int] = None,
) -> int:
    """
    Greedily complete *state* using PU1 (base PR3) or PU2 (base PR4).
    Returns the number of relocations used.
    """
    base = PR3 if mode == PU1 else PR4
    dst_func = _PR_DST_FUNCS[base]

    state, target = reduce_state(state, target, n_total)
    reloc = 0
    budget = node_budget or (n_total * max_tiers * len(state) * 2 + 32)

    while target <= n_total and reloc <= budget:
        loc = _locate(state, target)
        if loc is None:
            break
        ts, _ = loc
        p = state[ts][-1]

        if not _has_case_a_dst(state, ts, max_tiers, p):
            helper = _find_helper_move(state, ts, max_tiers)
            if helper is not None:
                j, k = helper
                state = _move_top(state, j, k)
                reloc += 1
                state, target = reduce_state(state, target, n_total)
                continue

        dst = dst_func(state, ts, max_tiers, p)
        if dst is None:
            break
        state = _move_top(state, ts, dst)
        reloc += 1
        state, target = reduce_state(state, target, n_total)

    return reloc


def probe_best_of(state: State, target: int, n_total: int, max_tiers: int) -> int:
    """"PU+": best of PR3, PR4 (as plain restricted-style probes, still
    legal unrestricted moves) and PU1, PU2."""
    from_pr = min(
        _probe_via_pr(state, target, n_total, max_tiers, PR3),
        _probe_via_pr(state, target, n_total, max_tiers, PR4),
    )
    from_pu = min(
        probe_unrestricted(state, target, n_total, max_tiers, PU1),
        probe_unrestricted(state, target, n_total, max_tiers, PU2),
    )
    return min(from_pr, from_pu)


def _probe_via_pr(state: State, target: int, n_total: int, max_tiers: int, mode: str) -> int:
    """Plain PR3/PR4 (only moves the current target's own blocker) --
    always a legal (if unexploited) unrestricted move sequence too."""
    dst_func = _PR_DST_FUNCS[mode]
    state, target = reduce_state(state, target, n_total)
    reloc = 0
    budget = n_total * max_tiers * len(state) + 16
    while target <= n_total and reloc <= budget:
        loc = _locate(state, target)
        if loc is None:
            break
        ts, _ = loc
        p = state[ts][-1]
        dst = dst_func(state, ts, max_tiers, p)
        if dst is None:
            break
        state = _move_top(state, ts, dst)
        reloc += 1
        state, target = reduce_state(state, target, n_total)
    return reloc


# ================================================================ #
#  IDA* (Algorithm 1 / Algorithm 2, unrestricted branching)          #
# ================================================================ #

class _Ctx:
    __slots__ = ("best", "nodes", "stop_event", "deadline", "aborted", "visited")

    def __init__(self, best: float, stop_event, deadline: float, use_map: bool) -> None:
        self.best       = best
        self.nodes      = 0
        self.stop_event = stop_event
        self.deadline   = deadline
        self.aborted    = False
        self.visited: Optional[Dict[State, int]] = {} if use_map else None


def _should_abort(ctx: _Ctx) -> bool:
    # Unlike the restricted variant, the unrestricted branching factor
    # (up to W*(W-1)) makes each *node* far more expensive (largely due
    # to the PU1/PU2 probe calls at frontier nodes), so the deadline is
    # checked much more frequently to avoid large overshoots.
    if ctx.aborted:
        return True
    if ctx.nodes % 16 == 0:
        if ctx.stop_event is not None and ctx.stop_event.is_set():
            ctx.aborted = True
            return True
        if time.perf_counter() > ctx.deadline:
            ctx.aborted = True
            return True
    return False


def _dfs(
    state: State, target: int, g: int, threshold: float,
    ctx: _Ctx, lb_func, n_total: int, max_tiers: int, probe_mode: str,
    last_move: Optional[Tuple[int, int]],
) -> float:
    ctx.nodes += 1
    if _should_abort(ctx):
        return float("inf")

    state, target = reduce_state(state, target, n_total)

    if target > n_total:
        if g < ctx.best:
            ctx.best = g
        return float("inf")

    if ctx.visited is not None:
        seen = ctx.visited.get(state)
        if seen is not None and seen <= g:
            return float("inf")
        ctx.visited[state] = g

    node_lb = g + lb_func(state, target, n_total, max_tiers)

    if node_lb > threshold:
        if node_lb < ctx.best and time.perf_counter() <= ctx.deadline:
            reloc = probe_unrestricted(state, target, n_total, max_tiers, probe_mode)
            total = g + reloc
            if total < ctx.best:
                ctx.best = total
        return node_lb

    if node_lb >= ctx.best:
        return float("inf")

    best_child_lb = float("inf")
    n_stacks = len(state)
    for src in range(n_stacks):
        if not state[src]:
            continue
        for dst in range(n_stacks):
            if dst == src or len(state[dst]) >= max_tiers:
                continue
            # Forbid immediately undoing the parent's move -- safe *only*
            # when there is no transposition table, since which child is
            # excluded depends on the incoming edge (`last_move`); with
            # the Map active, two different paths reaching the same state
            # would then have their "visited" slot consumed by whichever
            # path arrives first while excluding *different* children,
            # silently losing the other path's legitimate branch. The
            # Map already prevents the resulting back-and-forth cycles
            # on its own, so it is disabled here rather than combined.
            if (
                ctx.visited is None
                and last_move is not None
                and (src, dst) == (last_move[1], last_move[0])
            ):
                continue
            if ctx.aborted:
                break
            child = _move_top(state, src, dst)
            child_lb = _dfs(
                child, target, g + 1, threshold, ctx, lb_func,
                n_total, max_tiers, probe_mode, (src, dst),
            )
            if child_lb < best_child_lb:
                best_child_lb = child_lb

    return best_child_lb


def ida_star_unrestricted(
    state0: State,
    target0: int,
    n_total: int,
    max_tiers: int,
    lb_mode: str = LB1,
    probe_mode: str = PU2,
    use_map: bool = True,
    time_limit_s: float = 300.0,
    stop_event=None,
) -> Dict:
    """
    IDA*-U / IDA*-UM / IDA*-UM3 (Section VII-D):
      - ``lb_mode=LB1, use_map=False``  -> IDA*-U  (exact, no dedup)
      - ``lb_mode=LB1, use_map=True``   -> IDA*-UM (exact, transposition table)
      - ``lb_mode=LB3, use_map=True``   -> IDA*-UM3 (aggressive pruning,
        **not guaranteed optimal** -- ``optimal`` is always False)

    Returns a dict with keys ``best``, ``root_lb``, ``optimal``,
    ``nodes``, ``aborted`` (see ``ida_star_restricted`` in the CRP-R
    package for field semantics).
    """
    lb_func = _LB_FUNCS[lb_mode]
    inadmissible = (lb_mode == LB3)

    state0, target0 = reduce_state(state0, target0, n_total)
    if target0 > n_total:
        return dict(best=0, root_lb=0, optimal=True, nodes=1, aborted=False)

    initial_ub = probe_best_of(state0, target0, n_total, max_tiers)

    ctx = _Ctx(
        best=float(initial_ub),
        stop_event=stop_event,
        deadline=time.perf_counter() + time_limit_s,
        use_map=use_map,
    )
    threshold = float(lb_func(state0, target0, n_total, max_tiers))
    optimal = False

    while not ctx.aborted:
        if ctx.visited is not None:
            ctx.visited.clear()
        node_lb = _dfs(state0, target0, 0, threshold, ctx, lb_func, n_total, max_tiers, probe_mode, None)
        if ctx.aborted:
            break
        if ctx.best <= threshold:
            optimal = not inadmissible
            break
        if node_lb == float("inf"):
            optimal = not inadmissible
            break
        threshold = max(threshold + 1, node_lb)

    return dict(
        best=int(ctx.best),
        root_lb=int(threshold),
        optimal=optimal,
        nodes=ctx.nodes,
        aborted=ctx.aborted,
    )
