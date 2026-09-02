"""
Robust blocking-matrix search core for TierneyVoss2016RCPMP.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

Stacks = Dict[int, List[int]]  # stack_idx -> priorities bottom..top
Move = Tuple[int, int]         # (src, dst)
State = Tuple[Tuple[int, ...], ...]


def clone_stacks(stacks: Stacks) -> Stacks:
    return {k: list(v) for k, v in stacks.items()}


def to_state(stacks: Stacks) -> State:
    return tuple(tuple(stacks[s]) for s in sorted(stacks))


def from_state(state: State) -> Stacks:
    return {i: list(arr) for i, arr in enumerate(state)}


def apply_move_inplace(stacks: Stacks, src: int, dst: int, max_tiers: int) -> bool:
    if src == dst:
        return False
    if src not in stacks or dst not in stacks:
        return False
    if not stacks[src] or len(stacks[dst]) >= max_tiers:
        return False
    x = stacks[src].pop()
    stacks[dst].append(x)
    return True


def legal_moves(stacks: Stacks, max_tiers: int) -> List[Move]:
    out: List[Move] = []
    keys = list(stacks.keys())
    for src in keys:
        if not stacks[src]:
            continue
        for dst in keys:
            if src == dst:
                continue
            if len(stacks[dst]) < max_tiers:
                out.append((src, dst))
    return out


def build_blocking_matrix_from_intervals(
    intervals: Dict[int, Tuple[int, int]],
) -> Dict[Tuple[int, int], int]:
    """
    b(i, j)=1 if i blocks j (i may not be stacked above j), else 0.
    """
    ids = sorted(intervals.keys())
    b: Dict[Tuple[int, int], int] = {}
    for i in ids:
        si, ei = intervals[i]
        for j in ids:
            if i == j:
                b[(i, j)] = 1
                continue
            sj, ej = intervals[j]
            overlap = not (ei < sj or ej < si)
            if overlap:
                b[(i, j)] = 1
            elif si > ej:
                b[(i, j)] = 1
            else:
                b[(i, j)] = 0
    return b


def build_intervals_from_priorities(
    priorities: Iterable[int],
    robust_window: int,
) -> Dict[int, Tuple[int, int]]:
    ids = sorted(priorities)
    n = len(ids)
    out: Dict[int, Tuple[int, int]] = {}
    for p in ids:
        s = max(1, p - robust_window)
        e = min(n, p + robust_window)
        out[p] = (s, e)
    return out


def is_robust_sorted(stacks: Stacks, b: Dict[Tuple[int, int], int]) -> bool:
    for arr in stacks.values():
        for i in range(1, len(arr)):
            above = arr[i]
            for j in range(i):
                below = arr[j]
                if b.get((above, below), 0) == 1:
                    return False
    return True


def robust_blocking_containers_lb(stacks: Stacks, b: Dict[Tuple[int, int], int]) -> int:
    """
    Admissible lower bound: number of *distinct* containers that currently block
    at least one container below in their stack.
    """
    blockers: Set[int] = set()
    for arr in stacks.values():
        for i in range(1, len(arr)):
            above = arr[i]
            for j in range(i):
                below = arr[j]
                if b.get((above, below), 0) == 1:
                    blockers.add(above)
                    break
    return len(blockers)


def _topological_levels(
    nodes: Sequence[int],
    edges: Dict[int, Set[int]],
) -> Dict[int, int]:
    indeg = {v: 0 for v in nodes}
    for u in nodes:
        for w in edges[u]:
            indeg[w] += 1
    q = deque([v for v in nodes if indeg[v] == 0])
    lvl = {v: 1 for v in nodes}
    seen = 0
    while q:
        u = q.popleft()
        seen += 1
        for w in edges[u]:
            lvl[w] = max(lvl[w], lvl[u] + 1)
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    if seen == len(nodes):
        return lvl
    # Cycles should not occur for one-way constraints, but be safe.
    return {v: 1 for v in nodes}


def relaxation_groups_from_blocking_matrix(
    container_ids: Sequence[int],
    b: Dict[Tuple[int, int], int],
) -> Dict[int, int]:
    """
    Build a CPMP-style group assignment used as robust relaxation guidance.
    """
    ids = sorted(container_ids)

    # one-way blocking: if i blocks j and j does not block i -> group_i > group_j
    # edge j -> i
    edges: Dict[int, Set[int]] = {i: set() for i in ids}
    mutual_pairs: List[Tuple[int, int]] = []
    for i in ids:
        for j in ids:
            if i >= j:
                continue
            bij = b.get((i, j), 0)
            bji = b.get((j, i), 0)
            if bij == 1 and bji == 0:
                edges[j].add(i)
            elif bij == 0 and bji == 1:
                edges[i].add(j)
            elif bij == 1 and bji == 1:
                mutual_pairs.append((i, j))

    groups = _topological_levels(ids, edges)
    succ: Dict[int, Set[int]] = defaultdict(set)
    for u in ids:
        for w in edges[u]:
            succ[u].add(w)

    # enforce mutual inequality by bumping and propagating precedence.
    changed = True
    loops = 0
    while changed and loops < 10_000:
        loops += 1
        changed = False
        for i, j in mutual_pairs:
            if groups[i] != groups[j]:
                continue
            bump = max(i, j)
            groups[bump] += 1
            changed = True
            dq = deque([bump])
            while dq:
                u = dq.popleft()
                for w in succ[u]:
                    if groups[w] <= groups[u]:
                        groups[w] = groups[u] + 1
                        dq.append(w)
    return groups


def relaxed_sorted_lb(
    stacks: Stacks,
    groups: Dict[int, int],
) -> int:
    blockers: Set[int] = set()
    for arr in stacks.values():
        for i in range(1, len(arr)):
            g_above = groups[arr[i]]
            for j in range(i):
                if groups[arr[j]] < g_above:
                    blockers.add(arr[i])
                    break
    return len(blockers)


@dataclass
class SolveResult:
    moves: List[Move]
    solved: bool
    expanded_nodes: int
    lower_bound: int
    elapsed_s: float


def solve_rcpmp_ida_star(
    stacks_init: Stacks,
    max_tiers: int,
    blocking_matrix: Dict[Tuple[int, int], int],
    relax_groups: Dict[int, int],
    time_limit_s: float,
    depth_padding: int,
) -> SolveResult:
    start = time.perf_counter()
    expanded = 0

    def timed_out() -> bool:
        return (time.perf_counter() - start) >= time_limit_s

    init_lb = max(
        robust_blocking_containers_lb(stacks_init, blocking_matrix),
        relaxed_sorted_lb(stacks_init, relax_groups),
    )
    bound = init_lb
    max_bound = init_lb + max(0, depth_padding)

    init_state = to_state(stacks_init)
    path: List[Move] = []

    def dfs(
        state: State,
        g: int,
        bound_now: int,
        prev_move: Optional[Move],
        seen_depth: Dict[State, int],
    ) -> Tuple[bool, int]:
        nonlocal expanded
        if timed_out():
            return False, 10**9

        stacks = from_state(state)
        if is_robust_sorted(stacks, blocking_matrix):
            return True, g

        lb = max(
            robust_blocking_containers_lb(stacks, blocking_matrix),
            relaxed_sorted_lb(stacks, relax_groups),
        )
        f = g + lb
        if f > bound_now:
            return False, f

        expanded += 1
        nxt_threshold = 10**9

        moves = legal_moves(stacks, max_tiers)
        scored: List[Tuple[int, Move]] = []
        for mv in moves:
            src, dst = mv
            if prev_move is not None and mv == (prev_move[1], prev_move[0]):
                continue
            trial = clone_stacks(stacks)
            ok = apply_move_inplace(trial, src, dst, max_tiers)
            if not ok:
                continue
            sc = max(
                robust_blocking_containers_lb(trial, blocking_matrix),
                relaxed_sorted_lb(trial, relax_groups),
            )
            scored.append((sc, mv))
        scored.sort(key=lambda x: x[0])

        for _, mv in scored:
            trial = clone_stacks(stacks)
            ok = apply_move_inplace(trial, mv[0], mv[1], max_tiers)
            if not ok:
                continue
            s2 = to_state(trial)
            prev_best = seen_depth.get(s2)
            if prev_best is not None and prev_best <= g + 1:
                continue
            seen_depth[s2] = g + 1
            path.append(mv)
            found, res = dfs(s2, g + 1, bound_now, mv, seen_depth)
            if found:
                return True, res
            path.pop()
            nxt_threshold = min(nxt_threshold, res)
        return False, nxt_threshold

    solved = False
    while bound <= max_bound and not timed_out():
        found, nxt = dfs(init_state, 0, bound, None, {init_state: 0})
        if found:
            solved = True
            break
        if nxt >= 10**9:
            break
        bound = max(bound + 1, nxt)

    return SolveResult(
        moves=list(path),
        solved=solved,
        expanded_nodes=expanded,
        lower_bound=init_lb,
        elapsed_s=time.perf_counter() - start,
    )

