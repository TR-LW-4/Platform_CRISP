"""
GlahLayout – internal state representation for the GLAH algorithm.

Port of Layout.java + State.java from crp-glah-main (Bo Jin, 2015).

Design
------
This class is INTERNAL to the GLAH algorithm only.  It is NOT a
replacement for the platform's Yard/Stack/Container.  It exists because
GLAH needs fast O(1) queries:
  - support_capacity(s):  min priority in stack s  (= minUnderInclusive[s][height])
  - is_top_well_placed(s): top container is not badly placed
  - bad_count:            total badly-placed containers

These are maintained incrementally on every move / undo to avoid O(N)
rescans inside the lookahead tree.

Coordinate convention (1-indexed like the Java source):
  - stacks are numbered 1 … S
  - tiers  are numbered 1 … H  (tier 1 = bottom)
  - to == 0  means retrieval (container leaves the bay)

Conversion helpers:
  build_from_yard(yard, containers)  →  GlahLayout
  to_relocation_plan(ops, uid_to_container, S)  →  RelocationPlan
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

INF_PRI   = 10_000    # ≡ Constant.INF_PRIORITYLABEL
FLOOR_PRI = 500       # ≡ Constant.FLOOR_PRIORITYLABEL  (sentinel below tier 0)


# ================================================================ #
#  GlahContainer                                                    #
# ================================================================ #

class GlahContainer:
    """Lightweight mirror of Container.java."""
    __slots__ = ("priority", "uid")

    def __init__(self, priority: int, uid: int):
        self.priority = priority   # priorityLabel
        self.uid      = uid        # uniqueContainerIndex (1-based)

    def __repr__(self) -> str:
        return f"GC(pri={self.priority}, uid={self.uid})"


# ================================================================ #
#  GlahLayout                                                       #
# ================================================================ #

class GlahLayout:
    """
    Mutable bay state with O(1) support-capacity and undo support.

    Attributes mirroring Layout.java
    ---------------------------------
    S, H, N, G  – geometry / container counts
    bay[s][t]   – GlahContainer or None  (1-indexed s, t)
    height[s]   – current height of stack s
    min_under[s][t] – min priority in positions 1…t of stack s
                      (= minUnderInclusive[s][t] in Java)
    at_stack[uid], at_tier[uid] – current location of container uid
    container_groups[g] – sorted set of uids with priority g
    next_group   – the group (priority) of the next container to retrieve
    remain       – containers not yet retrieved
    bad_count    – number of badly-placed containers
    """

    def __init__(self, S: int, H: int, N: int, G: int):
        self.S = S
        self.H = H
        self.N = N
        self.G = G

        self.bay: List[List[Optional[GlahContainer]]] = [
            [None] * (H + 1) for _ in range(S + 1)
        ]
        self.height: List[int] = [0] * (S + 1)

        # min_under[s][t] = min priority in positions 1…t of stack s
        # min_under[s][0] = FLOOR_PRI (sentinel, > all real priorities)
        self.min_under: List[List[int]] = [
            [FLOOR_PRI] + [0] * H for _ in range(S + 1)
        ]

        self.at_stack: List[int] = [0] * (N + 1)   # uid → stack (negative = retrieved)
        self.at_tier:  List[int] = [0] * (N + 1)   # uid → tier

        # container_groups[g] = sorted list of uids with that priority
        self.container_groups: List[List[int]] = [[] for _ in range(G + 1)]
        self.next_group: int = 1
        self.remain:     int = N
        self.bad_count:  int = 0

    # ---------------------------------------------------------------- #
    # Factory                                                            #
    # ---------------------------------------------------------------- #

    @classmethod
    def build_from_yard(
        cls,
        yard,
        containers: list,
    ) -> "GlahLayout":
        """
        Convert a platform Yard + container list → GlahLayout.

        Platform stacks are keyed by (bay, row) tuples; GLAH uses
        flat integer indices 1…S.  We sort keys to get a stable mapping.
        """
        stack_keys = sorted(yard.stacks.keys())   # [(b,r), ...]
        S = len(stack_keys)
        H = yard.max_tiers
        N = sum(s.height for s in yard.stacks.values())

        # Build priority map: platform Container → (priority, uid)
        # Priorities are 1…N (CRP-R sequential assignment)
        all_containers = {c.id: c for c in containers}

        # Determine G = max priority group (for simplex = N)
        G = max(c.priority for c in containers) if containers else 1

        layout = cls(S, H, N, G)

        # stack index mapping: (bay,row) → 1-based GLAH stack index
        key_to_s: Dict[Tuple, int] = {k: (i + 1) for i, k in enumerate(stack_keys)}
        layout._key_to_s = key_to_s
        layout._s_to_key = {v: k for k, v in key_to_s.items()}

        uid = 0
        for key in stack_keys:
            s = key_to_s[key]
            stk = yard.stacks[key]
            for t_idx, plat_c in enumerate(stk.containers):
                t   = t_idx + 1
                uid += 1
                gc  = GlahContainer(priority=plat_c.priority, uid=uid)
                layout.bay[s][t]     = gc
                layout.height[s]     = t
                layout.at_stack[uid] = s
                layout.at_tier[uid]  = t

                # Update min_under
                layout.min_under[s][t] = min(plat_c.priority, layout.min_under[s][t - 1])

                # Badly placed: priority > min below (including self)
                if layout.min_under[s][t - 1] < plat_c.priority:
                    layout.bad_count += 1

                layout.container_groups[plat_c.priority].append(uid)

        # Sort each group's uid list
        for g in range(G + 1):
            layout.container_groups[g].sort()

        # Advance next_group past empty groups
        while (layout.next_group <= G and
               not layout.container_groups[layout.next_group]):
            layout.next_group += 1

        # Store reverse mapping: uid → platform container id  (for plan building)
        layout._uid_to_platid: Dict[int, int] = {}
        uid = 0
        for key in stack_keys:
            stk = yard.stacks[key]
            for plat_c in stk.containers:
                uid += 1
                layout._uid_to_platid[uid] = plat_c.id

        return layout

    def copy(self) -> "GlahLayout":
        other = GlahLayout.__new__(GlahLayout)
        other.S = self.S
        other.H = self.H
        other.N = self.N
        other.G = self.G
        other.bay = [col[:] for col in self.bay]
        other.height    = self.height[:]
        other.min_under = [row[:] for row in self.min_under]
        other.at_stack  = self.at_stack[:]
        other.at_tier   = self.at_tier[:]
        other.container_groups = [list(g) for g in self.container_groups]
        other.next_group = self.next_group
        other.remain     = self.remain
        other.bad_count  = self.bad_count
        if hasattr(self, "_key_to_s"):
            other._key_to_s    = self._key_to_s
            other._s_to_key    = self._s_to_key
            other._uid_to_platid = self._uid_to_platid
        return other

    # ---------------------------------------------------------------- #
    # Query helpers (mirror of Layout.java methods)                     #
    # ---------------------------------------------------------------- #

    def is_empty(self) -> bool:
        return self.remain == 0

    def top_container(self, s: int) -> Optional[GlahContainer]:
        t = self.height[s]
        return self.bay[s][t] if t > 0 else None

    def support_capacity(self, s: int) -> int:
        """min priority in the stack including top  (= minUnderInclusive[s][height])."""
        return self.min_under[s][self.height[s]]

    def support_capacity_except_top(self, s: int) -> int:
        t = self.height[s] - 1
        return self.min_under[s][t] if t >= 0 else FLOOR_PRI

    def is_badly_placed(self, s: int, t: int) -> bool:
        gc = self.bay[s][t]
        return gc is not None and gc.priority != self.min_under[s][t]

    def is_top_well_placed(self, s: int) -> bool:
        t = self.height[s]
        if t == 0:
            return True
        return not self.is_badly_placed(s, t)

    def contain_target(self, s: int) -> bool:
        """True if the stack contains a container of the next group to retrieve."""
        return self.next_group == self.support_capacity(s)

    # ---------------------------------------------------------------- #
    # Move / Undo (mirror of Layout.doMove / Layout.undoMove)           #
    # ---------------------------------------------------------------- #

    def do_move(self, from_s: int, to_s: int, gc: GlahContainer) -> None:
        """
        Execute one move.  to_s == 0 means retrieval.
        Caller must pass the correct gc (top of from_s).
        """
        uid = gc.uid
        t   = self.height[from_s]

        # ── remove from source ─────────────────────────────────── #
        if self.is_badly_placed(from_s, t):
            self.bad_count -= 1

        self.bay[from_s][t]      = None
        self.min_under[from_s][t] = 0
        self.height[from_s]      -= 1

        if to_s == 0:
            # retrieval
            self.remain            -= 1
            self.at_stack[uid]      = -self.at_stack[uid]
            self.at_tier[uid]       = -self.at_tier[uid]

            self.container_groups[gc.priority].remove(uid)
            while (self.next_group <= self.G and
                   not self.container_groups[self.next_group]):
                self.next_group += 1
        else:
            # relocation
            t2 = self.height[to_s] + 1
            self.height[to_s]       = t2
            self.bay[to_s][t2]      = gc
            prev_min                = self.min_under[to_s][t2 - 1]
            self.min_under[to_s][t2] = min(gc.priority, prev_min)
            self.at_stack[uid]      = to_s
            self.at_tier[uid]       = t2
            if self.is_badly_placed(to_s, t2):
                self.bad_count += 1

    def undo_move(self, from_s: int, to_s: int, gc: GlahContainer) -> None:
        """Undo a previously executed move (from_s, to_s, gc)."""
        uid = gc.uid
        if to_s == 0:
            # undo retrieval
            self.remain += 1
            self.container_groups[gc.priority].append(uid)
            self.container_groups[gc.priority].sort()
            self.next_group = gc.priority   # restored; may be lower than before
        else:
            # undo relocation: remove from to_s
            t2 = self.height[to_s]
            if self.is_badly_placed(to_s, t2):
                self.bad_count -= 1
            self.bay[to_s][t2]      = None
            self.min_under[to_s][t2] = 0
            self.height[to_s]       -= 1

        # restore to from_s
        t = self.height[from_s] + 1
        self.height[from_s]         = t
        self.bay[from_s][t]         = gc
        prev_min                    = self.min_under[from_s][t - 1]
        self.min_under[from_s][t]   = min(gc.priority, prev_min)
        self.at_stack[uid]          = from_s
        self.at_tier[uid]           = t
        if self.is_badly_placed(from_s, t):
            self.bad_count += 1


# ================================================================ #
#  GlahState                                                        #
# ================================================================ #

class GlahOp:
    """Single crane operation (relocation or retrieval)."""
    __slots__ = ("gc", "from_s", "to_s", "is_advice")

    def __init__(self, gc: GlahContainer, from_s: int, to_s: int,
                 is_advice: bool = False):
        self.gc       = gc
        self.from_s   = from_s
        self.to_s     = to_s         # 0 = retrieval
        self.is_advice = is_advice

    @property
    def is_retrieval(self) -> bool:
        return self.to_s == 0

    @property
    def is_relocation(self) -> bool:
        return self.to_s != 0

    def equal_to(self, other: "GlahOp") -> bool:
        return (self.gc.uid == other.gc.uid and
                self.from_s == other.from_s and
                self.to_s   == other.to_s)


class GlahState:
    """
    Mutable search state (layout + solution log + undo stack).
    Port of State.java.
    """

    def __init__(self, layout: GlahLayout):
        self.inst:            GlahLayout        = layout
        self.ops:             List[GlahOp]      = []   # solution log
        self.reloc_count:     int               = 0
        self.best_ops:        Optional[List[GlahOp]] = None
        self.best_reloc:      int               = INF_PRI
        self.probing_advices: Optional[List[GlahOp]] = None

    # ---------------------------------------------------------------- #
    # Move management                                                    #
    # ---------------------------------------------------------------- #

    def go_one_step(self, op: GlahOp) -> None:
        if op.is_advice:
            if self.probing_advices:
                self.probing_advices.pop(0)
        elif op.is_relocation:
            self.probing_advices = None

        self.ops.append(op)
        if op.is_relocation:
            self.reloc_count += 1
        self.inst.do_move(op.from_s, op.to_s, op.gc)

    def undo(self) -> None:
        op = self.ops.pop()
        if op.is_relocation:
            self.reloc_count -= 1
        self.inst.undo_move(op.from_s, op.to_s, op.gc)

    def size(self) -> int:
        return len(self.ops)

    def is_empty(self) -> bool:
        return self.inst.is_empty()

    # ---------------------------------------------------------------- #
    # Auto-retrieve                                                      #
    # ---------------------------------------------------------------- #

    def try_retrievals(self) -> int:
        """
        Auto-retrieve any accessible target containers.
        Returns the count of retrieved containers.
        """
        if self.is_empty():
            return 0

        pivot = _get_nearest_target(self.inst)
        if pivot is None:
            return 0
        uid   = pivot.uid
        count = 0

        while self.inst.at_tier[uid] == self.inst.height[self.inst.at_stack[uid]]:
            s  = self.inst.at_stack[uid]
            op = GlahOp(pivot, s, 0)
            self.go_one_step(op)
            count += 1

            if self.is_empty():
                break
            pivot = _get_nearest_target(self.inst)
            if pivot is None:
                break
            uid = pivot.uid

        return count

    def try_retrievals_from(self, s: int) -> int:
        """Auto-retrieve from a specific stack (used after vacating move)."""
        count = 0
        while (self.inst.height[s] > 0 and
               self.inst.top_container(s).priority == self.inst.next_group):
            gc  = self.inst.top_container(s)
            op  = GlahOp(gc, s, 0)
            self.go_one_step(op)
            count += 1
        return count

    # ---------------------------------------------------------------- #
    # Best solution tracking                                             #
    # ---------------------------------------------------------------- #

    def update_best(self) -> bool:
        reduced = _reduce_solution(self.ops, self.inst.N)
        n_reloc = sum(1 for op in reduced if op.is_relocation)
        if self.best_ops is None or self.best_reloc > n_reloc:
            self.best_ops   = reduced
            self.best_reloc = n_reloc
            return True
        return False


# ================================================================ #
#  Lower bound (Forster & Bortfeldt 2012 LB)                       #
# ================================================================ #

def lower_bound(inst: GlahLayout) -> int:
    """
    LBFB: bad_count + optional +1 (Forster-Bortfeldt correction).
    Port of LowerBound.LBFB.
    """
    # Simulate auto-retrievals on a copy to update bad_count
    sim = inst.copy()
    pivot = _get_nearest_target(sim)
    while pivot is not None:
        uid = pivot.uid
        s   = sim.at_stack[uid]
        if sim.at_tier[uid] == sim.height[s]:
            sim.do_move(s, 0, pivot)
            pivot = _get_nearest_target(sim)
        else:
            break

    bad = sim.bad_count

    # Forster–Bortfeldt +1 correction
    no_empty = True
    min_top  = INF_PRI
    max_min  = 0

    for s in range(1, sim.S + 1):
        if sim.height[s] == 0:
            no_empty = False
        else:
            top = sim.top_container(s)
            if top is not None:
                min_top = min(min_top, top.priority)
        max_min = max(max_min, sim.support_capacity(s))

    fb = 1 if (no_empty and min_top > max_min) else 0
    return bad + fb


# ================================================================ #
#  Urgent target selection                                          #
# ================================================================ #

def _get_nearest_target(inst: GlahLayout) -> Optional[GlahContainer]:
    """
    Return the urgent target: among all containers of inst.next_group,
    choose the one with fewest blockers above (Nearest_MinLargestAbove).
    Port of UrgentTargetSelection.getUrgentTarget(..., Nearest_MinLargestAbove).
    """
    if inst.is_empty():
        return None

    g = inst.next_group
    if g > inst.G or not inst.container_groups[g]:
        return None

    best_uid    = None
    best_dist   = INF_PRI
    best_largest = INF_PRI

    for uid in inst.container_groups[g]:
        s = inst.at_stack[uid]
        t = inst.at_tier[uid]

        # Java filter: skip if tier is too deep to matter
        # t < remain - (S-1)*H  →  skip
        if t < inst.remain - (inst.S - 1) * inst.H:
            continue

        dist = inst.height[s] - t

        largest_above = 0
        for jt in range(t + 1, inst.height[s] + 1):
            gc = inst.bay[s][jt]
            if gc is not None:
                largest_above = max(largest_above, gc.priority)

        if (best_uid is None or
                dist < best_dist or
                (dist == best_dist and largest_above < best_largest)):
            best_uid     = uid
            best_dist    = dist
            best_largest = largest_above

    if best_uid is None:
        # fallback: pick first in group
        uid = inst.container_groups[g][0]
        return inst.bay[inst.at_stack[uid]][inst.at_tier[uid]]

    uid = best_uid
    return inst.bay[inst.at_stack[uid]][inst.at_tier[uid]]


# ================================================================ #
#  Solution reducer (SolutionReducer.java)                         #
# ================================================================ #

def _reduce_solution(ops: List[GlahOp], N: int) -> List[GlahOp]:
    """
    Merge consecutive moves of the same container where possible.
    Port of SolutionReducer.reduce().
    """
    result:   List[GlahOp] = []
    last_op:  Dict[int, int] = {}   # uid → index in result

    for op in ops:
        if op.is_retrieval:
            result.append(op)
        else:
            uid = op.gc.uid
            if uid not in last_op:
                last_op[uid] = len(result)
                result.append(op)
            else:
                to2    = op.to_s
                pre_i  = last_op[uid]
                # Check if to2 appears in result[pre_i+1:]
                to2_appears = any(
                    (m.from_s == to2 or m.to_s == to2)
                    for m in result[pre_i + 1:]
                )
                if to2_appears:
                    last_op[uid] = len(result)
                    result.append(op)
                else:
                    # Combine: replace pre_i's op with merged (from_s→to2)
                    pre_op = result[pre_i]
                    combined = GlahOp(op.gc, pre_op.from_s, to2)
                    result[pre_i] = combined

    return result


# ================================================================ #
#  Plan builder                                                     #
# ================================================================ #

def ops_to_relocation_plan(
    ops:     List[GlahOp],
    layout:  GlahLayout,
) -> "RelocationPlan":
    """
    Convert a list of GlahOp into a platform RelocationPlan.

    Uses layout._uid_to_platid to map uid → platform container id,
    and layout._s_to_key to map GLAH stack index → (bay, row).
    """
    from core.plan import Movement, RelocationPlan

    plan = RelocationPlan()
    for op in ops:
        if op.is_retrieval:
            plan.add(Movement(
                container_id=layout._uid_to_platid[op.gc.uid],
                from_pos=layout._s_to_key[op.from_s],
                to_pos=None,
            ))
        else:
            plan.add(Movement(
                container_id=layout._uid_to_platid[op.gc.uid],
                from_pos=layout._s_to_key[op.from_s],
                to_pos=layout._s_to_key[op.to_s],
            ))
    return plan
