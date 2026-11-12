"""
Look-ahead probing for GLAHHeuristic.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple

from .layout import (
    GlahContainer, GlahLayout, GlahOp, GlahState,
    INF_PRI, lower_bound, _reduce_solution,
)
from .evaluate import evaluation_heuristic


# ================================================================ #
#  Relocation categories                                            #
# ================================================================ #

class Placement(IntEnum):
    FTB = 0   # Freeing Target, Badly placed source
    NTB = 1   # Non-freeing Target, Badly placed source
    G   = 2   # Well-placed (Good) source


def _classify_src(inst: GlahLayout, s: int) -> Placement:
    """Classify the source stack top."""
    if inst.is_top_well_placed(s):
        return Placement.G
    if inst.contain_target(s):
        return Placement.FTB
    return Placement.NTB


def _all_available_relocations(
    state:    GlahState,
    ftbg_n:  int,
    nfbg_n:  int,
    ftbb_n:  int,
    nfbb_n:  int,
    gg_n:    int,
    gb_n:    int,
) -> List[GlahOp]:
    """
    Port of Lookahead.allAvailableRelocations.
    Returns at most (ftbg_n + nfbg_n + ...) candidate relocations,
    one empty stack is treated as interchangeable (emptyTrial).
    """
    inst = state.inst

    ftbg: List[Tuple[int, GlahOp]] = []
    nfbg: List[Tuple[int, GlahOp]] = []
    ftbb: List[Tuple[int, GlahOp]] = []
    nfbb: List[Tuple[int, GlahOp]] = []
    gg:   List[Tuple[int, GlahOp]] = []
    gb:   List[Tuple[int, GlahOp]] = []

    for i in range(1, inst.S + 1):
        if inst.height[i] == 0:
            continue
        p1  = _classify_src(inst, i)
        con = inst.top_container(i)

        empty_trial = False
        for j in range(1, inst.S + 1):
            if j == i:
                continue
            if inst.height[j] >= inst.H:
                continue
            if inst.height[j] == 0:
                if empty_trial:
                    continue     # only one empty stack representative
                empty_trial = True

            # destination classification
            p2_good = (con.priority <= inst.support_capacity(j))

            op = GlahOp(con, i, j)

            if p1 == Placement.FTB and p2_good:
                score = inst.support_capacity(j) - con.priority
                ftbg.append((score, op))
            elif p1 == Placement.NTB and p2_good:
                score = inst.support_capacity(j) - con.priority
                nfbg.append((score, op))
            elif p1 == Placement.FTB and not p2_good:
                score = con.priority - inst.support_capacity(j)
                ftbb.append((score, op))
            elif p1 == Placement.NTB and not p2_good:
                score = con.priority - inst.support_capacity(j)
                nfbb.append((score, op))
            elif p2_good:   # p1 == G
                score = inst.support_capacity(j) - inst.support_capacity_except_top(i)
                gg.append((score, op))
            else:           # p1 == G, p2 bad
                score = -inst.support_capacity(j) - inst.support_capacity_except_top(i)
                gb.append((score, op))

    ftbg.sort(key=lambda x: x[0])
    nfbg.sort(key=lambda x: x[0])
    ftbb.sort(key=lambda x: x[0])
    nfbb.sort(key=lambda x: x[0])
    gg.sort(key=lambda x: x[0])
    gb.sort(key=lambda x: x[0])

    result: List[GlahOp] = []
    for score, op in ftbg[:ftbg_n]:
        result.append(op)
    for score, op in nfbg[:nfbg_n]:
        result.append(op)
    for score, op in ftbb[:ftbb_n]:
        result.append(op)
    for score, op in nfbb[:nfbb_n]:
        result.append(op)
    for score, op in gg[:gg_n]:
        result.append(op)
    for score, op in gb[:gb_n]:
        result.append(op)
    return result


# ================================================================ #
#  Hash for cycle detection                                         #
# ================================================================ #

def _hash_state(inst: GlahLayout, depth: int) -> int:
    """Port of Lookahead.hashValue – hash of bay config + depth."""
    parts = [str(depth)]
    for s in range(1, inst.S + 1):
        parts.append("[")
        parts.append(",".join(
            str(inst.bay[s][t].priority)
            for t in range(1, inst.height[s] + 1)
        ))
        parts.append("]")
    return hash("".join(parts))


# ================================================================ #
#  Tree search                                                      #
# ================================================================ #

class Lookahead:
    """
    Port of Lookahead.java.

    Parameters
    ----------
    depth_limit : D in the paper (default 3)
    ftbg_n … gb_n : max candidates per category (paper defaults: 5,5,3,3,1,1)
    """

    def __init__(
        self,
        depth_limit: int = 3,
        ftbg_n:  int = 5,
        nfbg_n:  int = 5,
        ftbb_n:  int = 3,
        nfbb_n:  int = 3,
        gg_n:    int = 1,
        gb_n:    int = 1,
    ):
        self.depth_limit = depth_limit
        self.ftbg_n = ftbg_n
        self.nfbg_n = nfbg_n
        self.ftbb_n = ftbb_n
        self.nfbb_n = nfbb_n
        self.gg_n   = gg_n
        self.gb_n   = gb_n
        self._visited: Set[int] = set()

    def most_promising_relocation(self, state: GlahState) -> Optional[GlahOp]:
        """Return the best next relocation, or None if none found."""
        self._visited = set()
        try:
            result = self._get_best_branch(state, 0)
            return result[0] if result is not None else None
        except Exception:
            return None

    def _get_best_branch(
        self,
        state: GlahState,
        depth: int,
    ) -> Optional[Tuple[Optional[GlahOp], int]]:
        """
        Port of Lookahead.getBestBranch.
        Returns (best_op, best_reloc_count) or None if pruned.
        """
        h = _hash_state(state.inst, depth)
        if h in self._visited:
            return None
        self._visited.add(h)

        lb_now = lower_bound(state.inst)
        if lb_now + state.reloc_count >= state.best_reloc:
            return None

        backup = state.size()

        # Leaf node: evaluate heuristic
        if state.is_empty() or depth >= self.depth_limit:
            saved_advices = state.probing_advices

            if not state.is_empty():
                evaluation_heuristic(state)

            cost = state.reloc_count
            while state.size() > backup:
                state.undo()
            state.probing_advices = saved_advices

            return (None, cost)

        # Branch node
        best: Optional[Tuple[Optional[GlahOp], int]] = None

        candidates = _all_available_relocations(
            state,
            self.ftbg_n, self.nfbg_n,
            self.ftbb_n, self.nfbb_n,
            self.gg_n, self.gb_n,
        )

        # Add probe advice (first move from evaluation heuristic)
        if state.probing_advices is None:
            evaluation_heuristic(state)
            temp: List[GlahOp] = []
            while state.size() > backup:
                last_op = state.ops[-1]
                if last_op.is_relocation:
                    temp.insert(0, GlahOp(last_op.gc, last_op.from_s, last_op.to_s,
                                          is_advice=True))
                state.undo()
            state.probing_advices = temp

        saved_advices = state.probing_advices

        if state.probing_advices:
            prob_op = state.probing_advices[0]
            # Prepend probe advice (avoid duplicate if already in candidates)
            candidates = [prob_op] + [
                c for c in candidates if not c.equal_to(prob_op)
            ]

        for op in candidates:
            state.go_one_step(op)
            state.try_retrievals()

            child = self._get_best_branch(state, depth + 1)

            if child is not None:
                _, child_cost = child
                if best is None or best[1] > child_cost:
                    best = (op, child_cost)

            while state.size() > backup:
                state.undo()
            state.probing_advices = saved_advices

            if op.is_advice and state.probing_advices:
                state.probing_advices.insert(0, op)

        return best
