"""
Multi-bay retrieval of Forster & Bortfeldt (Oper. Res. Proc. 2011), Sec. 3.2:
each bay is emptied by its own move sequence (the C&OR tree search, p. 260),
and the bay sequences are merged into one block sequence by Algorithm 1
(p. 261). Also an independent replay check of a block sequence and the
conversion to crane_time positions. Decisions P1-P14:
docs/validation/forster_bortfeldt_2011_multibay.md §2.

A bay move is (d, r, g) as in state.py (stacks from 0, r None for a remove).
A block move is (src, dst, g) with src = (bay, stack), dst = (bay, stack) or
None for a remove, bays and stacks from 0.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite
F. Forster, A. Bortfeldt, "A tree search heuristic for the container
retrieval problem", Operations Research Proceedings 2011, 257-262 (2012).
--------------------------------------------------------------------------
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .crane_time import Position
from .search import Params, TreeSearch
from .state import Bay, Move

Slot = Tuple[int, int]
BlockMove = Tuple[Slot, Optional[Slot], int]
Block = Sequence[Sequence[Sequence[int]]]


@dataclass(frozen=True)
class BayRun:
    """One bay search: its moves in block groups, wall clock (s), and how it ended."""
    moves: List[Move]
    seconds: float
    timed_out: bool
    optimal: bool
    nodes: int
    lower_bound: int
    greedy_moves: int


def ranked(stacks: Sequence[Sequence[int]]) -> Tuple[List[List[int]], Dict[int, int]]:
    """P12: the bay's groups renumbered 1..n, order kept; returns the layout and rank -> block group."""
    groups = sorted({g for s in stacks for g in s})
    rank = {g: i + 1 for i, g in enumerate(groups)}
    return [[rank[g] for g in s] for s in stacks], {i + 1: g for i, g in enumerate(groups)}


def solve_bay(stacks: Sequence[Sequence[int]], H: int, params: Params = Params(),
              clock: Callable[[], float] = time.perf_counter) -> BayRun:
    """One bay as an independent CRP instance (p. 260): groups ranked (P12), G = containers in the bay (P13),
    the C&OR search with its own time limit (P10, P14)."""
    n = sum(map(len, stacks))
    if n == 0:
        return BayRun([], 0.0, False, True, 0, 0, 0)
    layout, block_group = ranked(stacks)
    t0 = clock()
    ts = TreeSearch(Bay.from_stacks(layout, H, G=n), params, clock)
    plan = ts.run()
    seconds = clock() - t0
    return BayRun([(d, r, block_group[g]) for d, r, g in plan], seconds, ts.timed_out, ts.optimal, ts.nodes,
                  ts.lb_root, ts.greedy_moves)


def solve_block(bays: Block, H: int, params: Params = Params(),
                clock: Callable[[], float] = time.perf_counter) -> Tuple[List[BlockMove], List[BayRun]]:
    """Every bay searched on its own, one after another in index order (P11), then merged (Algorithm 1)."""
    runs = [solve_bay(bay, H, params, clock) for bay in bays]
    return merge(bays, [r.moves for r in runs]), runs


def merge(bays: Block, solutions: Sequence[Sequence[Move]]) -> List[BlockMove]:
    """Algorithm 1 (p. 261): while the block is not empty, for every bay (in index order) whose lowest group is
    the block's lowest group, append that bay's next moves up to and including its next remove."""
    layout = [[list(s) for s in bay] for bay in bays]
    queues = [deque(sol) for sol in solutions]
    if len(queues) != len(layout):
        raise ValueError("one solution per bay is needed")
    block: List[BlockMove] = []
    while any(s for bay in layout for s in bay):
        low_block = min(g for bay in layout for s in bay for g in s)
        for b, bay in enumerate(layout):
            if not any(bay) or min(g for s in bay for g in s) != low_block:
                continue
            removed = False
            while not removed:
                if not queues[b]:
                    raise ValueError(f"bay {b}: its solution ends before a remove of group {low_block}")
                d, r, g = queues[b].popleft()
                if not bay[d] or bay[d][-1] != g:
                    raise ValueError(f"bay {b}: item {g} is not on top of stack {d}")
                item = bay[d].pop()
                if r is None:
                    removed = True
                else:
                    bay[r].append(item)
                block.append(((b, d), None if r is None else (b, r), g))
    if any(queues):
        raise ValueError("moves left in a bay solution after the block is empty")
    return block


def check_block_plan(bays: Block, H: int, plan: Sequence[BlockMove]) -> int:
    """Replays plan on plain lists: only top items move, a remove takes an item of the lowest group in the block,
    no stack exceeds H, and the block is empty at the end; raises ValueError on the first violation.
    Moves between bays are allowed. Returns the number of relocations."""
    layout = [[list(s) for s in bay] for bay in bays]
    if any(len(s) > H for bay in layout for s in bay):
        raise ValueError("initial layout higher than H")
    relocations = 0
    for i, ((b, d), dst, g) in enumerate(plan):
        if not (0 <= b < len(layout) and 0 <= d < len(layout[b])) or not layout[b][d]:
            raise ValueError(f"move {i}: source {(b, d)} empty or unknown")
        if layout[b][d][-1] != g:
            raise ValueError(f"move {i}: item {g} is not on top of {(b, d)}")
        if dst is None:
            if g != min(x for bay in layout for s in bay for x in s):
                raise ValueError(f"move {i}: remove of {g} out of group order")
            layout[b][d].pop()
            continue
        rb, r = dst
        if not (0 <= rb < len(layout) and 0 <= r < len(layout[rb])) or dst == (b, d):
            raise ValueError(f"move {i}: bad target {dst}")
        if len(layout[rb][r]) >= H:
            raise ValueError(f"move {i}: stack {dst} is full (H = {H})")
        layout[rb][r].append(layout[b][d].pop())
        relocations += 1
    if any(s for bay in layout for s in bay):
        raise ValueError("block not empty at the end")
    return relocations


def to_positions(plan: Sequence[BlockMove]) -> List[Tuple[Position, Optional[Position]]]:
    """Block moves -> crane_time positions (bay, row), both from 1 (P5)."""
    return [((b + 1, d + 1), None if dst is None else (dst[0] + 1, dst[1] + 1)) for (b, d), dst, _g in plan]
