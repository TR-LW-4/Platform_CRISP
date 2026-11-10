"""
The tree search of Forster & Bortfeldt (2012), Sec. 6 (p. 303-306): the greedy
initial solution (Fig. 4), the productive moves (Fig. 7), the compound moves
(Fig. 6) and the depth-first search over them (Fig. 5). The objective is the
number of moves, equivalently relocations (p. 306); relocations are
unrestricted (p. 302). Decisions F1-F19: docs/validation/forster_bortfeldt_2012.md §2.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite
F. Forster, A. Bortfeldt, "A tree search procedure for the container
relocation problem", Computers & Operations Research 39 (2012) 299-309.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from .state import Bay, Move


@dataclass(frozen=True)
class Params:
    """Parameter setting of p. 306."""
    n_succ: int = 10
    cm_stop_threshold: int = 150
    max_flg_bb: int = 3
    max_gg: int = 2
    time_limit: Optional[float] = 60.0


def greedy(bay: Bay) -> List[Move]:
    """find_initial_solution (Fig. 4, p. 303); ties go to the first move in the order of Bay.relocations (F8)."""
    s: List[Move] = []
    while bay.n:
        rm = bay.removes()
        if rm:
            for m in rm:
                s.append(m)
                bay = bay.apply(m)
            continue
        bg = bge = bb_flg = None
        for m, before, after, flg in bay.relocations():
            d, r, g = m
            target = bay.stacks[r]
            if before and not after and target:
                key = target[-1] - g                       # F10
                if bg is None or key < bg[0]:
                    bg = (key, m)
            elif before and not after:
                if bge is None or g < bge[0]:
                    bge = (g, m)
            elif before and after and flg:
                key = g - bay.mins[r][-1]                  # F9: minimum, against the printed maximum
                if bb_flg is None or key < bb_flg[0]:
                    bb_flg = (key, m)
        best = bg or bge or bb_flg
        if best is None:
            raise RuntimeError("no move of the four priority groups of Fig. 4 is applicable")
        s.append(best[1])
        bay = bay.apply(best[1])
    return s


def productive_moves(bay: Bay, p: Params) -> List[Move]:
    """determine_productive_moves (Fig. 7, p. 306)."""
    rm = bay.removes()
    if rm:
        return rm
    bg, flg_bb, gg = [], [], []
    for m, before, after, flg in bay.relocations():
        d, r, g = m
        if before and not after:
            bg.append(m)
        elif before and after and flg:
            flg_bb.append((g - bay.mins[r][-1], m))
        elif not before and not after:
            sd, sr = bay.stacks[d], bay.stacks[r]
            below = sd[-2] if len(sd) > 1 else bay.G     # F12
            top = sr[-1] if sr else bay.G
            gg.append(((g - below) - (g - top), m))      # F11: diffBefore - diffAfter, signed
    if bg:
        return bg
    flg_bb.sort(key=lambda x: x[0])                      # stable: ties in generation order (F17)
    gg.sort(key=lambda x: x[0])
    return [m for _k, m in flg_bb[:p.max_flg_bb]] + [m for _k, m in gg[:p.max_gg]]


class TreeSearch:
    """perform_compound_moves (Fig. 5) and determine_compound_moves (Fig. 6), p. 304-305."""

    def __init__(self, bay: Bay, params: Params = Params(), clock: Callable[[], float] = time.perf_counter):
        self.root, self.p, self.clock = bay, params, clock
        self.best: List[Move] = []
        self.timed_out = False
        self.nodes = 0

    def run(self) -> List[Move]:
        """Greedy, then the search; returns the best move sequence s*. The time limit includes the greedy (F15)."""
        self.t0 = self.clock()
        self.lb_root = self.root.lower_bound()
        self.best = greedy(self.root)
        self.greedy_moves = len(self.best)
        limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(limit, 20 * self.root.n + 1000))
        try:
            self._perform([], self.root)
        finally:
            sys.setrecursionlimit(limit)
        return self.best

    @property
    def optimal(self) -> bool:
        return len(self.best) == self.lb_root

    def _out_of_time(self) -> bool:
        if self.p.time_limit is not None and self.clock() - self.t0 > self.p.time_limit:
            self.timed_out = True
        return self.timed_out

    def _perform(self, s: List[Move], bay: Bay) -> None:
        if self.optimal or self._out_of_time():
            return
        self.nodes += 1
        if bay.n == 0:
            if len(s) < len(self.best):
                self.best = s
            return
        cm: List[tuple] = []
        self._determine(bay, s, [], len(self.best) - len(s) - 1, 1, cm)
        cm.sort(key=lambda c: (len(c[0]) + c[1].lower_bound(), -len(c[0]), -c[1].clean_supply()))
        for moves, nxt in cm[:self.p.n_succ]:
            self._perform(s + moves, nxt)

    def _determine(self, bay: Bay, s: List[Move], cm_work: List[Move], max_moves: int,
                   cm_stop_value: int, cm: List[tuple]) -> None:
        pm = productive_moves(bay, self.p)
        for m in pm:
            nxt = bay.apply(m)
            if nxt.n == 0:
                if len(s) + len(cm_work) + 1 < len(self.best):
                    self.best = s + cm_work + [m]
                continue
            if nxt.lower_bound() > max_moves:              # F14: as printed in Fig. 6
                continue
            work = cm_work + [m]
            if cm_stop_value < self.p.cm_stop_threshold:
                self._determine(nxt, s, work, max_moves - 1, cm_stop_value * len(pm), cm)
            else:
                cm.append((work, nxt))
