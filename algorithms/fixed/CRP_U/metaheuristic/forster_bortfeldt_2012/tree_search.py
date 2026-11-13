"""
Compound-move tree search core for ForsterBortfeldt2012.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

from .layout import FBLayout

# ---------------------------------------------------------------------------
# Type alias: a single crane operation
# ---------------------------------------------------------------------------
Move = Tuple   # ('rem', s) | ('rel', d, r)


# ================================================================ #
#  Helper: count moves                                              #
# ================================================================ #

def _count_relocations(ops: List[Move]) -> int:
    return sum(1 for op in ops if op[0] == "rel")


def _apply_move(lay: FBLayout, mv: Move) -> None:
    if mv[0] == "rem":
        lay.apply_remove(mv[1])
    else:
        lay.apply_relocation(mv[1], mv[2])


# ================================================================ #
#  Productive-move set (§6.4 determine_productive_moves)           #
# ================================================================ #

def _productive_moves(
    lay: FBLayout,
    max_flg_bb: int,
    max_gg: int,
) -> List[Move]:
    """
    Return the ordered list of productive moves for the current layout.

    Follows Fig. 7: removes dominate → BG → (FLG_BB ∪ GG).
    When removes are present, ALL of them are returned.
    When BG relocations are present, ALL of them are returned.
    FLG_BB and GG are each capped at max_flg_bb / max_gg and sorted.
    """
    S = lay.S
    H = lay.H

    # ── Step 1: removes ────────────────────────────────────────────── #
    removes: List[Move] = [
        ("rem", s)
        for s in range(S)
        if lay.stacks[s] and lay.stacks[s][-1] == lay.next_group
    ]
    if removes:
        return removes

    # ── Step 2: BG relocations ──────────────────────────────────────── #
    bg: List[Move] = []
    for d in range(S):
        stk_d = lay.stacks[d]
        if not stk_d or not lay.is_top_badly_placed(d):
            continue
        g_item = stk_d[-1]
        for r in range(S):
            if r == d or len(lay.stacks[r]) >= H:
                continue
            # Well-placed at destination: gmin(r) >= g_item  OR  r is empty
            if not lay.stacks[r] or lay.gmin(r) >= g_item:
                bg.append(("rel", d, r))
    if bg:
        return bg

    # ── Step 3: FLG_BB ─────────────────────────────────────────────── #
    flg_bb_scored: List[Tuple] = []
    for d in range(S):
        stk_d = lay.stacks[d]
        if not stk_d or not lay.is_top_badly_placed(d):
            continue
        g_item = stk_d[-1]
        # FLG: stack d contains a next_group item below the top
        if lay.next_group not in stk_d[:-1]:
            continue
        for r in range(S):
            if r == d or len(lay.stacks[r]) >= H:
                continue
            stk_r = lay.stacks[r]
            # BB condition: gmin(r) < g_item  (stays badly placed at dest)
            if stk_r and lay.gmin(r) < g_item:
                # Sort ascending by (g_item − gmin_r):
                # prefer smaller difference to avoid early re-relocation
                flg_bb_scored.append((g_item - lay.gmin(r), d, r))
    flg_bb_scored.sort()
    flg_bb: List[Move] = [("rel", m[1], m[2]) for m in flg_bb_scored[:max_flg_bb]]

    # ── Step 4: GG relocations ─────────────────────────────────────── #
    gg_scored: List[Tuple] = []
    for d in range(S):
        stk_d = lay.stacks[d]
        if not stk_d or lay.is_top_badly_placed(d):
            continue   # source must be well-placed
        g_item = stk_d[-1]
        for r in range(S):
            if r == d or len(lay.stacks[r]) >= H:
                continue
            stk_r = lay.stacks[r]
            # GG: well-placed at dest (gmin_r >= g_item  OR  r empty)
            if stk_r and lay.gmin(r) < g_item:
                continue
            # Sort ascending by (diffBefore − diffAfter):
            #   diffBefore = g_item − group_directly_below_in_d
            #   diffAfter  = g_item − gtop_of_r
            diff_before = g_item - stk_d[-2] if len(stk_d) >= 2 else 0
            diff_after  = g_item - stk_r[-1]  if stk_r          else 0
            gg_scored.append((diff_before - diff_after, d, r))
    gg_scored.sort()
    gg: List[Move] = [("rel", m[1], m[2]) for m in gg_scored[:max_gg]]

    return flg_bb + gg


# ================================================================ #
#  Greedy initial solution (§6.1 find_initial_solution, Fig. 4)   #
# ================================================================ #

def greedy_solution(
    layout: FBLayout,
    max_flg_bb: int = 3,
    max_gg: int = 2,
) -> Tuple[List[Move], int]:
    """
    Build a greedy solution for layout (Fig. 4).

    Priority order at each step:
      1. Execute ALL removes in Rm (loop until none possible).
      2. BG to non-empty stack  → min(gtop_r − g_item).
      3. BG to empty stack      → min(g_item).
      4. FLG_BB                 → max(g_item − gmin_r).

    Returns (ops, n_relocations).
    """
    lay = layout.copy()
    ops: List[Move] = []
    S = lay.S
    H = lay.H

    while not lay.is_empty():
        # ── Step 1: removes ─────────────────────────────────────────── #
        did_remove = True
        while did_remove:
            did_remove = False
            for s in range(S):
                stk = lay.stacks[s]
                if stk and stk[-1] == lay.next_group:
                    lay.apply_remove(s)
                    ops.append(("rem", s))
                    did_remove = True
                    if lay.is_empty():
                        break
            if lay.is_empty():
                break
        if lay.is_empty():
            break

        # ── Step 2: BG to non-empty ──────────────────────────────────── #
        bg_nonempty: List[Tuple] = []
        for d in range(S):
            if not lay.stacks[d] or not lay.is_top_badly_placed(d):
                continue
            g_item = lay.stacks[d][-1]
            for r in range(S):
                if r == d or len(lay.stacks[r]) >= H or not lay.stacks[r]:
                    continue
                if lay.gmin(r) >= g_item:
                    # difference = gtop(r) − g_item (minimize)
                    bg_nonempty.append((lay.gtop(r) - g_item, d, r))
        if bg_nonempty:
            bg_nonempty.sort()
            _, d, r = bg_nonempty[0]
            lay.apply_relocation(d, r)
            ops.append(("rel", d, r))
            continue

        # ── Step 3: BG to empty ──────────────────────────────────────── #
        bg_empty: List[Tuple] = []
        for d in range(S):
            if not lay.stacks[d] or not lay.is_top_badly_placed(d):
                continue
            g_item = lay.stacks[d][-1]
            for r in range(S):
                if r == d or len(lay.stacks[r]) >= H or lay.stacks[r]:
                    continue   # r must be empty
                bg_empty.append((g_item, d, r))
        if bg_empty:
            bg_empty.sort()
            _, d, r = bg_empty[0]
            lay.apply_relocation(d, r)
            ops.append(("rel", d, r))
            continue

        # ── Step 4: FLG_BB ───────────────────────────────────────────── #
        flg_bb: List[Tuple] = []
        for d in range(S):
            stk_d = lay.stacks[d]
            if not stk_d or not lay.is_top_badly_placed(d):
                continue
            g_item = stk_d[-1]
            if lay.next_group not in stk_d[:-1]:
                continue   # not FLG
            for r in range(S):
                if r == d or len(lay.stacks[r]) >= H:
                    continue
                gmin_r = lay.gmin(r)
                if lay.stacks[r] and gmin_r < g_item:   # BB condition
                    flg_bb.append((-(g_item - gmin_r), d, r))   # negate → max sort
        if flg_bb:
            flg_bb.sort()
            _, d, r = flg_bb[0]
            lay.apply_relocation(d, r)
            ops.append(("rel", d, r))
            continue

        # Fallback: any non-full stack (degenerate layout — should not occur
        # in well-formed instances but prevents infinite loops)
        for d in range(S):
            if not lay.stacks[d]:
                continue
            for r in range(S):
                if r == d or len(lay.stacks[r]) >= H:
                    continue
                lay.apply_relocation(d, r)
                ops.append(("rel", d, r))
                break
            else:
                continue
            break
        else:
            break   # stuck — exit to avoid infinite loop

    return ops, _count_relocations(ops)


# ================================================================ #
#  Tree search state                                                #
# ================================================================ #

class _TSState:
    """Mutable shared state threaded through all recursive calls."""

    __slots__ = (
        "best_ops", "best_n_total",
        "start_time", "time_limit",
        "n_succ", "cm_stop_threshold", "max_flg_bb", "max_gg",
        "lb_optimal",
    )

    def __init__(
        self,
        greedy_ops:   List[Move],
        greedy_total: int,
        initial_n:    int,
        time_limit:   float,
        n_succ:       int,
        cm_stop_threshold: int,
        max_flg_bb:   int,
        max_gg:       int,
    ) -> None:
        self.best_ops          = greedy_ops
        self.best_n_total      = greedy_total
        self.start_time        = time.monotonic()
        self.time_limit        = time_limit
        self.n_succ            = n_succ
        self.cm_stop_threshold = cm_stop_threshold
        self.max_flg_bb        = max_flg_bb
        self.max_gg            = max_gg
        # Absolute optimum = initial_n removes + 0 relocations
        self.lb_optimal        = initial_n

    def timed_out(self) -> bool:
        return time.monotonic() - self.start_time > self.time_limit

    def optimal_found(self) -> bool:
        return self.best_n_total <= self.lb_optimal


# ================================================================ #
#  Compound move generation (§6.3 determine_compound_moves, Fig. 6) #
# ================================================================ #

def _determine_compound_moves(
    lay:            FBLayout,
    cmwork:         List[Move],
    max_moves:      int,
    cm_stop_value:  int,
    prefix:         List[Move],
    state:          _TSState,
    cm_list:        List[List[Move]],
) -> None:
    """
    Recursive compound-move generator (Fig. 6).

    Appends completed compound moves to cm_list.
    When an empty layout is reached, updates state.best immediately.

    Parameters
    ----------
    lay           : current layout.
    cmwork        : moves accumulated in the current compound move.
    max_moves     : remaining move budget (must improve incumbent).
    cm_stop_value : compound-move-length controller; stops extending
                    when cm_stop_value >= state.cm_stop_threshold.
    prefix        : ops committed at the perform_compound_moves level
                    (needed to reconstruct full solution on empty-leaf).
    state         : shared mutable state.
    cm_list       : output: accumulates completed compound moves.
    """
    if state.timed_out():
        return

    pm_list = _productive_moves(lay, state.max_flg_bb, state.max_gg)
    if not pm_list:
        return

    for pm in pm_list:
        if state.timed_out():
            return

        new_lay = lay.copy()
        _apply_move(new_lay, pm)

        if new_lay.is_empty():
            # Complete solution — update best immediately (Fig. 6 §6.3)
            full_ops = prefix + cmwork + [pm]
            n_total  = len(full_ops)
            if n_total < state.best_n_total:
                state.best_ops     = full_ops
                state.best_n_total = n_total
            continue

        # Prune if lower bound already exceeds remaining budget
        if new_lay.lower_bound() > max_moves:
            continue

        new_cmwork    = cmwork + [pm]
        new_cm_stop   = cm_stop_value * len(pm_list)

        if new_cm_stop < state.cm_stop_threshold:
            # Keep extending the compound move
            _determine_compound_moves(
                new_lay, new_cmwork, max_moves - 1,
                new_cm_stop, prefix, state, cm_list,
            )
        else:
            # Compound move is long enough — add to candidates
            cm_list.append(new_cmwork)


# ================================================================ #
#  Tree search (§6.2 perform_compound_moves, Fig. 5)               #
# ================================================================ #

def _perform_compound_moves(
    lay:         FBLayout,
    current_ops: List[Move],
    state:       _TSState,
) -> None:
    """
    Recursive tree search (Fig. 5).

    current_ops : moves executed from the root to this node.
    """
    # ── Abort conditions ──────────────────────────────────────────── #
    if state.timed_out() or state.optimal_found():
        return

    if lay.is_empty():
        n = len(current_ops)
        if n < state.best_n_total:
            state.best_ops     = current_ops[:]
            state.best_n_total = n
        return

    max_moves = state.best_n_total - len(current_ops) - 1
    if max_moves < 0:
        return

    # ── Generate candidate compound moves ─────────────────────────── #
    cm_list: List[List[Move]] = []
    _determine_compound_moves(
        lay, [], max_moves, 1, current_ops, state, cm_list,
    )
    if not cm_list:
        return

    # ── Sort compound moves (Fig. 5) ──────────────────────────────── #
    # Primary   : ascending  len(cm) + lb(resulting layout)
    # Secondary : descending len(cm)          (tie-break)
    # Tertiary  : descending clean_supply of resulting layout
    def _sort_key(cm: List[Move]) -> Tuple:
        res = lay.copy()
        for mv in cm:
            _apply_move(res, mv)
        lb = res.lower_bound()
        cs = res.clean_supply()
        return (len(cm) + lb, -len(cm), -cs)

    cm_list.sort(key=_sort_key)

    # ── Recurse into top n_succ compound moves ────────────────────── #
    for cm in cm_list[: state.n_succ]:
        if state.timed_out() or state.optimal_found():
            break

        new_lay = lay.copy()
        for mv in cm:
            _apply_move(new_lay, mv)
        new_ops = current_ops + cm

        # Pre-prune: total so far + lb of new layout must beat incumbent
        if len(new_ops) + new_lay.lower_bound() >= state.best_n_total:
            continue

        _perform_compound_moves(new_lay, new_ops, state)


# ================================================================ #
#  Public entry point                                               #
# ================================================================ #

def run_tree_search(
    layout:            FBLayout,
    time_limit:        float = 60.0,
    n_succ:            int   = 5,
    cm_stop_threshold: int   = 150,
    max_flg_bb:        int   = 3,
    max_gg:            int   = 2,
) -> Tuple[List[Move], int]:
    """
    Run the Forster & Bortfeldt (2012) tree search on *layout*.

    Phase 1 – Greedy initial solution (§6.1).
    Phase 2 – Heuristic tree search with compound moves (§6.2 – §6.4).

    Parameters
    ----------
    layout            : initial bay layout.
    time_limit        : wall-clock seconds before stopping (paper: 60 s).
    n_succ            : max successors per tree node, nSucc (paper: 10).
    cm_stop_threshold : compound-move depth controller (paper: 150).
    max_flg_bb        : max FLG_BB productive moves per step (paper: 3).
    max_gg            : max GG productive moves per step (paper: 2).

    Returns
    -------
    (ops, n_relocations)
      ops            : full move list including removes and relocations.
      n_relocations  : number of relocation moves in ops.
    """
    # Phase 1 – greedy
    greedy_ops, _ = greedy_solution(layout, max_flg_bb, max_gg)
    greedy_total  = len(greedy_ops)

    state = _TSState(
        greedy_ops        = greedy_ops,
        greedy_total      = greedy_total,
        initial_n         = layout.n,
        time_limit        = time_limit,
        n_succ            = n_succ,
        cm_stop_threshold = cm_stop_threshold,
        max_flg_bb        = max_flg_bb,
        max_gg            = max_gg,
    )

    # Phase 2 – tree search
    _perform_compound_moves(layout.copy(), [], state)

    best_ops   = state.best_ops or greedy_ops
    n_relocs   = _count_relocations(best_ops)
    return best_ops, n_relocs
