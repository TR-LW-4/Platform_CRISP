"""
Abstraction + PDB search engine for KuArthanari2016Abstraction.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import heapq
import itertools
import time
from typing import Callable, Dict, Iterator, List, Optional, Tuple

State = Tuple[Tuple[int, ...], ...]

PDBLevels = Dict[int, Dict[State, int]]


# ================================================================ #
#  Generic state helpers (self-contained -- see module docstring)   #
# ================================================================ #

def state_from_stacks(stacks: List[List[int]]) -> State:
    """Build a ``State`` from plain ``List[List[int]]`` (bottom -> top)."""
    return tuple(tuple(s) for s in stacks)


def _locate(state: State, value: int) -> Optional[Tuple[int, int]]:
    """Return ``(stack_idx, height_idx)`` of *value*, or ``None``."""
    for si, stack in enumerate(state):
        for hi, v in enumerate(stack):
            if v == value:
                return si, hi
    return None


def reduce_state(state: State, target: int, n_total: int) -> Tuple[State, int]:
    """
    Convert *state* to its minimum equivalent layout: repeatedly retrieve
    *target* (for free) while it sits on top of its stack, advancing
    *target* each time.  Returns ``(new_state, new_target)``.
    """
    stacks = list(state)
    while target <= n_total:
        loc = _locate(tuple(stacks), target)
        if loc is None:
            target += 1
            continue
        si, hi = loc
        if hi != len(stacks[si]) - 1:
            break  # target is blocked
        stacks[si] = stacks[si][:-1]
        target += 1
    return tuple(stacks), target


def is_solved(state: State) -> bool:
    return all(len(s) == 0 for s in state)


def eligible_dsts(state: State, exclude_idx: int, max_tiers: int) -> List[int]:
    return [i for i, s in enumerate(state) if i != exclude_idx and len(s) < max_tiers]


def move_top(state: State, src: int, dst: int) -> State:
    stacks = list(state)
    v = stacks[src][-1]
    stacks[src] = stacks[src][:-1]
    stacks[dst] = stacks[dst] + (v,)
    return tuple(stacks)


# ================================================================ #
#  Abstraction mapping phi (Section 3.4)                            #
# ================================================================ #

def canonical_sort(cols) -> State:
    """Drop empty columns; sort the rest ascending by base (bottom) value."""
    non_empty = [tuple(c) for c in cols if c]
    non_empty.sort(key=lambda c: c[0])
    return tuple(non_empty)


def abstract_key(state: State) -> State:
    """
    phi(s) (Definition 2 / Section 3.4): relabel the remaining
    containers to contiguous ranks 1..r -- preserving their relative
    retrieval order -- drop empty columns, and sort columns ascending
    by base-slot rank.  Used both as the forward node-cache key
    (``CC_n``) and the PDB index, so structurally identical layouts
    collide regardless of which physical containers/columns realise
    them.
    """
    values = sorted(v for col in state for v in col)
    rank = {v: i + 1 for i, v in enumerate(values)}
    relabelled = [tuple(rank[v] for v in col) for col in state]
    return canonical_sort(relabelled)


# ================================================================ #
#  Pattern database construction (Section 4.3)                      #
# ================================================================ #

def _gen_canonical_states(ranks: Tuple[int, ...], m: int, cols_left: int) -> Iterator[State]:
    """
    Enumerate every canonical abstract state that places all of *ranks*
    into <= ``cols_left`` columns of height <= ``m``.

    Physically, a container's position within its column is *not*
    constrained by its priority value at all (e.g. column ``(4, 3)`` --
    priority 4 at the bottom, 3 on top -- is a perfectly valid, if
    inefficient, layout): there is no "smallest value must be at a
    column's base" invariant to exploit for a duplicate-free recursion.
    This enumerates every raw assignment of ranks to <= ``cols_left``
    slots (subject to the height cap), every within-slot ordering, then
    canonicalises (Section 3.4: drop empty columns, sort ascending by
    base value) and dedupes.  Correct by construction; adequate for the
    modest depths this PDB targets (the paper itself notes ~10-15 units
    is the practical ceiling on typical hardware -- Section 4.2).
    """
    seen = set()
    r = len(ranks)
    if r == 0:
        yield ()
        return

    n_slots = min(cols_left, r)
    for assignment in itertools.product(range(n_slots), repeat=r):
        groups: List[List[int]] = [[] for _ in range(n_slots)]
        feasible = True
        for elem, slot in zip(ranks, assignment):
            groups[slot].append(elem)
            if len(groups[slot]) > m:
                feasible = False
                break
        if not feasible:
            continue
        nonempty_groups = [g for g in groups if g]
        for perms in itertools.product(*(itertools.permutations(g) for g in nonempty_groups)):
            key = canonical_sort(perms)
            if key not in seen:
                seen.add(key)
                yield key


def _build_pdb_level(
    k: int,
    n_cols: int,
    m_tiers: int,
    pdb_prev: Dict[State, int],
    deadline: Optional[float] = None,
    stop_event=None,
) -> Optional[Dict[State, int]]:
    """
    Build PDB_k (Section 4.3) given the completed PDB_{k-1} table.

    Every legal action from a k-unit canonical state either
      (a) exposes rank 1 for a *free* retrieval -- this state's value is
          already known: pop rank 1, relabel 2..k -> 1..k-1, canonicalize,
          and look up PDB_{k-1}; or
      (b) is a same-level relocation (the restricted variant only ever
          allows moving the container on top of rank 1's own column) to
          another column, weight 1, staying at k units.
    This is a shortest-path problem over the k-unit state graph with a
    set of pre-seeded terminal values (a); solved by multi-source
    Dijkstra over the *reverse* graph, seeded from every terminal state.
    Returns ``None`` if the deadline / stop_event fires before this
    level's graph is fully constructed.
    """
    if k == 0:
        return {(): 0}

    ranks = tuple(range(1, k + 1))
    reverse_adj: Dict[State, List[State]] = {}
    heap: List[Tuple[int, State]] = []
    seen_states = 0

    for state in _gen_canonical_states(ranks, m_tiers, n_cols):
        seen_states += 1
        if seen_states % 4096 == 0:
            if stop_event is not None and stop_event.is_set():
                return None
            if deadline is not None and time.perf_counter() > deadline:
                return None

        reverse_adj.setdefault(state, [])
        src_idx = 0
        for si, col in enumerate(state):
            if 1 in col:
                src_idx = si
                break

        if state[src_idx][-1] == 1:
            # Rank 1 already on top -- free retrieval into PDB_{k-1}.
            popped = state[src_idx][:-1]
            cols = list(state)
            cols[src_idx] = popped
            successor_raw = canonical_sort(cols)
            succ_key = abstract_key(successor_raw)
            term_value = pdb_prev.get(succ_key, 0)
            heapq.heappush(heap, (term_value, state))
            continue

        blocker = state[src_idx][-1]
        new_src = state[src_idx][:-1]
        for dst_idx, col in enumerate(state):
            if dst_idx == src_idx or len(col) >= m_tiers:
                continue
            cols = list(state)
            cols[src_idx] = new_src
            cols[dst_idx] = col + (blocker,)
            succ = canonical_sort(cols)
            reverse_adj.setdefault(succ, []).append(state)
        if len(state) < n_cols:
            cols = list(state)
            cols[src_idx] = new_src
            cols.append((blocker,))
            succ = canonical_sort(cols)
            reverse_adj.setdefault(succ, []).append(state)

    dist: Dict[State, int] = {}
    while heap:
        d, u = heapq.heappop(heap)
        if u in dist:
            continue
        dist[u] = d
        for p in reverse_adj.get(u, ()):
            if p not in dist:
                heapq.heappush(heap, (d + 1, p))

    # Defensive: any canonical state not reached from a terminal (should
    # not happen for n_cols >= 2) falls back to its confirmed-relocations
    # lower bound so lookups never silently under-count.
    for state in reverse_adj:
        if state not in dist:
            dist[state] = _confirmed_relocations(state)

    return dist


def build_pdb(
    depth: int,
    n_cols: int,
    m_tiers: int,
    time_limit_s: Optional[float] = None,
    stop_event=None,
) -> PDBLevels:
    """
    Build PDB_0 .. PDB_``depth`` (Section 4.3).  Returns
    ``{r: {abstract_state: optimal_cost}}``.  Stops early (returning
    whatever levels completed) if ``time_limit_s`` elapses or
    ``stop_event`` is set -- callers should treat missing levels as "no
    PDB shortcut available" rather than an error.
    """
    deadline = None if time_limit_s is None else time.perf_counter() + time_limit_s
    levels: PDBLevels = {0: {(): 0}}
    for k in range(1, depth + 1):
        if deadline is not None and time.perf_counter() > deadline:
            break
        if stop_event is not None and stop_event.is_set():
            break
        level = _build_pdb_level(k, n_cols, m_tiers, levels[k - 1], deadline, stop_event)
        if level is None:
            break
        levels[k] = level
    return levels


# ================================================================ #
#  Lower bound / initial upper bound                                 #
# ================================================================ #

def _confirmed_relocations(state: State) -> int:
    """
    "Confirmed relocations" lower bound (Kim & Hong 2006, Section 2.1 /
    Zhu et al. 2012's LB1): count containers sitting above a
    smaller-priority container in the same stack.  Reimplemented locally
    per this platform's independence convention.
    """
    total = 0
    for stack in state:
        min_below: Optional[int] = None
        for v in stack:
            if min_below is not None and v > min_below:
                total += 1
            min_below = v if min_below is None else min(min_below, v)
    return total


def _greedy_probe(state: State, target: int, n_total: int, max_tiers: int) -> int:
    """
    Cheap initial upper bound (self-contained "PR1"-style probe, per
    Section 3.2's requirement for a starting incumbent): always relocate
    to the currently shortest eligible column.
    """
    state, target = reduce_state(state, target, n_total)
    reloc = 0
    guard = n_total * max_tiers * max(len(state), 1) + 16
    while target <= n_total and reloc <= guard:
        loc = _locate(state, target)
        if loc is None:
            break
        ts, _ = loc
        elig = eligible_dsts(state, ts, max_tiers)
        if not elig:
            break
        dst = min(elig, key=lambda i: (len(state[i]), i))
        state = move_top(state, ts, dst)
        reloc += 1
        state, target = reduce_state(state, target, n_total)
    return reloc


# ================================================================ #
#  Bidirectional depth-first branch-and-bound (Section 4)            #
# ================================================================ #

class _SearchContext:
    __slots__ = (
        "n_total", "max_tiers", "best", "nodes",
        "cache", "cache_depth", "max_cache_size", "cache_hits",
        "pdb_levels", "pdb_depth", "pdb_hits",
        "stop_event", "deadline", "aborted",
    )

    def __init__(
        self,
        n_total: int,
        max_tiers: int,
        pdb_levels: PDBLevels,
        pdb_depth: int,
        cache_depth: int,
        max_cache_size: int,
        stop_event,
        deadline: float,
    ) -> None:
        self.n_total = n_total
        self.max_tiers = max_tiers
        self.best = float("inf")
        self.nodes = 0
        self.cache: Dict[State, int] = {}
        self.cache_depth = cache_depth
        self.max_cache_size = max_cache_size
        self.cache_hits = 0
        self.pdb_levels = pdb_levels
        self.pdb_depth = pdb_depth
        self.pdb_hits = 0
        self.stop_event = stop_event
        self.deadline = deadline
        self.aborted = False


def _should_abort(ctx: _SearchContext) -> bool:
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


def _dfs(state: State, target: int, g: int, depth: int, ctx: _SearchContext) -> None:
    """
    Bidirectional DFBnB (Fig. 5): forward search with ``CC_n`` node
    caching, cut short by an exact PDB lookup (backward search) once the
    remaining container count is within the PDB's depth.
    """
    ctx.nodes += 1
    if _should_abort(ctx):
        return

    state, target = reduce_state(state, target, ctx.n_total)
    remaining = ctx.n_total - target + 1

    if remaining <= 0:
        if g < ctx.best:
            ctx.best = g
        return

    key: Optional[State] = None

    # ── Backward search: exact PDB shortcut (Section 3.3 / Fig. 5) ── #
    if remaining <= ctx.pdb_depth and remaining in ctx.pdb_levels:
        key = abstract_key(state)
        cost = ctx.pdb_levels[remaining].get(key)
        if cost is not None:
            ctx.pdb_hits += 1
            total = g + cost
            if total < ctx.best:
                ctx.best = total
            return

    # ── Lower-bound pruning ─────────────────────────────────────── #
    lb = g + _confirmed_relocations(state)
    if lb >= ctx.best:
        return

    # ── Forward search: CC_n node caching (Section 3.4 / Fig. 3) ──── #
    if depth <= ctx.cache_depth and len(ctx.cache) < ctx.max_cache_size:
        if key is None:
            key = abstract_key(state)
        cached_g = ctx.cache.get(key)
        if cached_g is not None and cached_g <= g:
            ctx.cache_hits += 1
            return
        ctx.cache[key] = g

    ts, _ = _locate(state, target)
    if ts is None:
        return  # defensive: should not happen

    children: List[Tuple[int, State]] = []
    for dst in eligible_dsts(state, ts, ctx.max_tiers):
        child = move_top(state, ts, dst)
        child_lb = g + 1 + _confirmed_relocations(child)
        children.append((child_lb, child))
    if not children:
        return  # degenerate instance: no feasible relocation

    children.sort(key=lambda t: t[0])

    for child_lb, child in children:
        if ctx.aborted:
            return
        if child_lb >= ctx.best:
            continue  # prune before descending
        _dfs(child, target, g + 1, depth + 1, ctx)


def solve(
    initial_stacks: List[List[int]],
    n_total: int,
    max_tiers: int,
    pdb_levels: Optional[PDBLevels] = None,
    pdb_depth: int = 0,
    cache_depth: int = 0,
    max_cache_size: int = 0,
    time_limit_s: float = 300.0,
    stop_event=None,
) -> Dict:
    """
    Run the Ku & Arthanari (2016) bidirectional abstraction search on
    *initial_stacks*.  Returns a dict with keys ``best`` (relocations of
    the best/optimal solution found), ``root_lb`` (confirmed-relocations
    lower bound at the root), ``optimal`` (True iff the search completed
    -- i.e. proved optimality -- before the time limit), ``nodes``
    (nodes explored), ``cache_hits``, ``pdb_hits``, ``aborted``.
    """
    pdb_levels = pdb_levels or {0: {(): 0}}
    state0 = state_from_stacks(initial_stacks)
    state0, target0 = reduce_state(state0, 1, n_total)
    remaining0 = n_total - target0 + 1

    if remaining0 <= 0:
        return dict(
            best=0, root_lb=0, optimal=True, nodes=1,
            cache_hits=0, pdb_hits=0, aborted=False,
        )

    initial_ub = _greedy_probe(state0, target0, n_total, max_tiers)
    root_lb = _confirmed_relocations(state0)

    ctx = _SearchContext(
        n_total=n_total,
        max_tiers=max_tiers,
        pdb_levels=pdb_levels,
        pdb_depth=pdb_depth,
        cache_depth=cache_depth,
        max_cache_size=max(1, max_cache_size),
        stop_event=stop_event,
        deadline=time.perf_counter() + time_limit_s,
    )
    ctx.best = float(initial_ub)

    _dfs(state0, target0, 0, 0, ctx)

    best = int(ctx.best) if ctx.best < float("inf") else -1
    optimal = (not ctx.aborted) and ctx.best < float("inf")

    return dict(
        best=best,
        root_lb=int(root_lb),
        optimal=optimal,
        nodes=ctx.nodes,
        cache_hits=ctx.cache_hits,
        pdb_hits=ctx.pdb_hits,
        aborted=ctx.aborted,
    )
