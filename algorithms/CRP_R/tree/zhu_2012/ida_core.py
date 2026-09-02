"""
IDA* search engine for Zhu2012IDAStarR.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

State = Tuple[Tuple[int, ...], ...]

PR1, PR2, PR3, PR4 = "PR1", "PR2", "PR3", "PR4"
LB1, LB2, LB3 = "LB1", "LB2", "LB3"


# ================================================================ #
#  State helpers                                                     #
# ================================================================ #

def state_from_stacks(stacks: List[List[int]]) -> State:
    """Build a ``State`` from plain ``List[List[int]]`` (bottom -> top)."""
    return tuple(tuple(s) for s in stacks)


def _locate(state: State, priority: int) -> Optional[Tuple[int, int]]:
    """Return ``(stack_idx, height_idx)`` of *priority*, or ``None``."""
    for si, stack in enumerate(state):
        for hi, p in enumerate(stack):
            if p == priority:
                return si, hi
    return None


def reduce_state(state: State, target: int, n_total: int) -> Tuple[State, int]:
    """
    Convert *state* to its minimum equivalent layout (Section V):
    repeatedly retrieve *target* while it sits on top of its stack,
    advancing *target* each time.  Returns ``(new_state, new_target)``.
    """
    stacks = list(state)
    while target <= n_total:
        loc = _locate(state, target)
        if loc is None:
            target += 1
            continue
        si, hi = loc
        if hi != len(stacks[si]) - 1:
            break  # target is blocked
        stacks[si] = stacks[si][:-1]
        state = tuple(stacks)
        target += 1
    return state, target


def is_solved(state: State) -> bool:
    return all(len(s) == 0 for s in state)


def _eligible_dsts(state: State, exclude_idx: int, max_tiers: int) -> List[int]:
    return [
        i for i, s in enumerate(state)
        if i != exclude_idx and len(s) < max_tiers
    ]


def _min_priority(stack: Tuple[int, ...]) -> float:
    return min(stack) if stack else float("inf")


def _move_top(state: State, src: int, dst: int) -> State:
    stacks = list(state)
    p = stacks[src][-1]
    stacks[src] = stacks[src][:-1]
    stacks[dst] = stacks[dst] + (p,)
    return tuple(stacks)


# ================================================================ #
#  Lower bounds (Section V-A)                                       #
# ================================================================ #

def _has_safe_dst(state: State, exclude_idx: int, p: int, max_tiers: int) -> bool:
    """
    True iff some stack other than *exclude_idx* can currently hold *p*
    without creating a priority inversion (Eq. 2's "safe destination"
    test): the stack is empty, or its current minimum priority > p.
    """
    for i, stack in enumerate(state):
        if i == exclude_idx or len(stack) >= max_tiers:
            continue
        if not stack or min(stack) > p:
            return True
    return False


def lb1(state: State, target: int = 0, n_total: int = 0, max_tiers: int = 0) -> int:
    """
    LB1 (Eq. 1) -- a.k.a. Kim & Hong's "confirmed relocations": count
    containers sitting above a smaller-priority container in the same
    stack.  Signature accepts (and ignores) the extra arguments so it
    is interchangeable with ``lb2``/``lb3``.
    """
    total = 0
    for stack in state:
        min_below: Optional[int] = None
        for p in stack:
            if min_below is not None and p > min_below:
                total += 1
            min_below = p if min_below is None else min(min_below, p)
    return total


def lb2(state: State, target: int, n_total: int = 0, max_tiers: int = 0) -> int:
    """
    LB1 + Eq. (2): every container above the *current target* (in the
    target's own stack) that has no safe destination stack right now
    is guaranteed at least one further relocation.  Admissible for the
    restricted variant only.
    """
    total = lb1(state)
    loc = _locate(state, target)
    if loc is None:
        return total
    ts, th = loc
    above = state[ts][th + 1:]
    for p in above:
        if not _has_safe_dst(state, ts, p, max_tiers):
            total += 1
    return total


def lb3(state: State, target: int, n_total: int, max_tiers: int) -> int:
    """
    LB1 + Eqs. (3)-(4): extend LB2's "no safe destination" test from
    only the target's own stack to *every* stack's local blockers
    (containers above that stack's own smallest remaining priority).

    For stack *s* with local minimum ``t' = min(s)``, containers above
    ``t'`` in *s* are tested against the *reduced* layout ``L'`` where
    every container with priority < t' (elsewhere in the yard) has
    already been removed, together with everything currently stacked
    above it -- modelling the yard state once t' becomes the active
    global target.  For the target's own stack, t' == target and this
    reduction is a no-op, so LB3 restricted to that stack reproduces
    LB2 exactly; LB3 = LB2 + (non-negative contributions from every
    other stack), hence LB3 >= LB2 >= LB1.  Admissible for the
    restricted variant only.
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
            # Removing a container with priority < t_prime requires first
            # removing everything *above* it (higher index = higher tier);
            # keep only the portion strictly below the lowest such
            # container (cascades correctly for multiple qualifying
            # containers in the same stack).
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


_LB_FUNCS: Dict[str, Callable[[State, int, int, int], int]] = {
    LB1: lb1, LB2: lb2, LB3: lb3,
}


# ================================================================ #
#  Probe heuristics PR1-PR4 (Section V-B.1)                          #
# ================================================================ #

def _pr1_dst(state: State, src: int, max_tiers: int, p: int) -> Optional[int]:
    elig = _eligible_dsts(state, src, max_tiers)
    if not elig:
        return None
    return min(elig, key=lambda i: (len(state[i]), i))


def _pr2_dst(state: State, src: int, max_tiers: int, p: int) -> Optional[int]:
    elig = _eligible_dsts(state, src, max_tiers)
    if not elig:
        return None
    return max(elig, key=lambda i: (_min_priority(state[i]), -i))


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


_PR_DST_FUNCS: Dict[str, Callable[[State, int, int, int], Optional[int]]] = {
    PR1: _pr1_dst, PR2: _pr2_dst, PR3: _pr3_dst, PR4: _pr4_dst,
}


def probe_restricted(
    state: State,
    target: int,
    n_total: int,
    max_tiers: int,
    mode: str,
    node_budget: Optional[int] = None,
) -> int:
    """
    Greedily complete *state* using probe heuristic *mode* (PR1-PR4).
    Returns the number of relocations used (independent of any already
    "confirmed" cost -- caller adds ``g``).  ``node_budget`` bounds the
    number of relocations attempted as a defensive guard against
    non-terminating heuristics on infeasible layouts.
    """
    dst_func = _PR_DST_FUNCS[mode]
    state, target = reduce_state(state, target, n_total)
    reloc = 0
    budget = node_budget or (n_total * max_tiers * len(state) + 16)

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


def probe_best_of(
    state: State, target: int, n_total: int, max_tiers: int,
    modes: Tuple[str, ...] = (PR1, PR2, PR3, PR4),
) -> int:
    """"PR+": the best solution found among several probe heuristics."""
    return min(
        probe_restricted(state, target, n_total, max_tiers, m)
        for m in modes
    )


# ================================================================ #
#  IDA* (Algorithm 1 / Algorithm 2)                                  #
# ================================================================ #

class _Ctx:
    __slots__ = ("best", "nodes", "stop_event", "deadline", "aborted")

    def __init__(self, best: float, stop_event, deadline: float) -> None:
        self.best       = best
        self.nodes      = 0
        self.stop_event = stop_event
        self.deadline   = deadline
        self.aborted    = False


def _should_abort(ctx: _Ctx) -> bool:
    if ctx.aborted:
        return True
    if ctx.nodes % 256 == 0:
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
) -> float:
    """
    Depth-first branch-and-bound (Algorithm 2).  Returns the smallest
    lower bound encountered among nodes beyond the current *threshold*
    in this subtree (used to seed the next IDA* iteration).
    """
    ctx.nodes += 1
    if _should_abort(ctx):
        return float("inf")

    state, target = reduce_state(state, target, n_total)

    if target > n_total:
        if g < ctx.best:
            ctx.best = g
        return float("inf")

    node_lb = g + lb_func(state, target, n_total, max_tiers)

    if node_lb > threshold:
        if node_lb < ctx.best:
            reloc = probe_restricted(state, target, n_total, max_tiers, probe_mode)
            total = g + reloc
            if total < ctx.best:
                ctx.best = total
        return node_lb

    if node_lb >= ctx.best:
        return float("inf")

    ts, _ = _locate(state, target)
    best_child_lb = float("inf")
    for dst in _eligible_dsts(state, ts, max_tiers):
        if ctx.aborted:
            break
        child = _move_top(state, ts, dst)
        child_lb = _dfs(child, target, g + 1, threshold, ctx, lb_func, n_total, max_tiers, probe_mode)
        if child_lb < best_child_lb:
            best_child_lb = child_lb

    return best_child_lb


def ida_star_restricted(
    state0: State,
    target0: int,
    n_total: int,
    max_tiers: int,
    lb_mode: str = LB3,
    probe_mode: str = PR4,
    time_limit_s: float = 300.0,
    stop_event=None,
) -> Dict:
    """
    IDA*-R (Section VII-C's best configuration by default):
    ``IDAStar(pi, PR+, LB3, PR4)``.

    Returns a dict with keys: ``best`` (relocations of the best
    solution found), ``root_lb`` (best proven lower bound), ``optimal``
    (True iff ``best == root_lb``), ``nodes`` (nodes explored),
    ``aborted`` (True iff the time limit was hit before proving
    optimality).
    """
    lb_func = _LB_FUNCS[lb_mode]

    state0, target0 = reduce_state(state0, target0, n_total)
    if target0 > n_total:
        return dict(best=0, root_lb=0, optimal=True, nodes=1, aborted=False)

    initial_ub = probe_best_of(state0, target0, n_total, max_tiers)

    ctx = _Ctx(
        best=float(initial_ub),
        stop_event=stop_event,
        deadline=time.perf_counter() + time_limit_s,
    )
    threshold = float(lb_func(state0, target0, n_total, max_tiers))
    optimal = False

    while not ctx.aborted:
        node_lb = _dfs(state0, target0, 0, threshold, ctx, lb_func, n_total, max_tiers, probe_mode)
        if ctx.aborted:
            break
        if ctx.best <= threshold:
            optimal = True
            break
        if node_lb == float("inf"):
            optimal = True
            break
        threshold = max(threshold + 1, node_lb)

    return dict(
        best=int(ctx.best),
        root_lb=int(threshold),
        optimal=optimal,
        nodes=ctx.nodes,
        aborted=ctx.aborted,
    )
