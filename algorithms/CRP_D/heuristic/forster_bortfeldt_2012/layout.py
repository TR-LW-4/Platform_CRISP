"""
Internal bay-state representation for Forster & Bortfeldt (2012).

Completely self-contained: no imports from other platform algorithms.

Reference
---------
F. Forster and A. Bortfeldt,
"A tree search procedure for the container relocation problem",
Computers & Operations Research 39 (2012) 299–309.

Convention
----------
stacks[s] : list of group integers, bottom→top (index 0 = bottom).
Group index: 1 … G.  Lower value = higher retrieval priority (retrieved first).
S          : number of stacks.
H          : maximum stack height (hard cap).
G          : maximum group index present in initial layout.
n          : current total items in bay.
next_group : current minimum group remaining (the "targets" to remove now).
"""

from __future__ import annotations

from typing import List


class FBLayout:
    """
    Mutable internal bay state.

    All mutation methods update n and next_group automatically.
    """

    __slots__ = ("stacks", "S", "H", "G", "n", "next_group")

    def __init__(
        self,
        stacks: List[List[int]],
        S: int,
        H: int,
        G: int,
        n: int,
        next_group: int,
    ) -> None:
        self.stacks     = stacks
        self.S          = S
        self.H          = H
        self.G          = G
        self.n          = n
        self.next_group = next_group

    # ------------------------------------------------------------------ #
    # Copy / status                                                         #
    # ------------------------------------------------------------------ #

    def copy(self) -> "FBLayout":
        return FBLayout(
            [list(s) for s in self.stacks],
            self.S, self.H, self.G, self.n, self.next_group,
        )

    def is_empty(self) -> bool:
        return self.n == 0

    # ------------------------------------------------------------------ #
    # Stack queries                                                          #
    # ------------------------------------------------------------------ #

    def gmin(self, s: int) -> int:
        """Minimum group in stack s; G+1 if empty."""
        stk = self.stacks[s]
        return min(stk) if stk else self.G + 1

    def gtop(self, s: int) -> int:
        """Group of topmost item in stack s; G+1 if empty."""
        stk = self.stacks[s]
        return stk[-1] if stk else self.G + 1

    def is_top_badly_placed(self, s: int) -> bool:
        """
        True iff the top item of stack s is badly placed.

        An item at the top is badly placed when some item below it has a
        strictly lower group index (= must be retrieved sooner), i.e. the
        top item is blocking that lower-priority item.
        """
        stk = self.stacks[s]
        if len(stk) < 2:
            return False
        g_top = stk[-1]
        return any(g < g_top for g in stk[:-1])

    # ------------------------------------------------------------------ #
    # Bay-level queries                                                      #
    # ------------------------------------------------------------------ #

    def nb(self) -> int:
        """Count of all badly placed items in the bay (n'_BG from §5)."""
        count = 0
        for stk in self.stacks:
            if len(stk) < 2:
                continue
            running_min = stk[0]
            for i in range(1, len(stk)):
                g = stk[i]
                if running_min < g:
                    count += 1
                if g < running_min:
                    running_min = g
        return count

    def lower_bound(self) -> int:
        """
        Forster & Bortfeldt (2012) §5 lower bound on total moves.

          n'_m = n'_REM + n'_BG + n'_non_BG
               = n     + nb   + n'_non_BG

        Returns lower bound on (removes + relocations).
        """
        return self.n + self.nb() + self._n_non_bg()

    def lower_bound_relocations(self) -> int:
        """Lower bound on relocations only: nb + n'_non_BG."""
        return self.nb() + self._n_non_bg()

    def _n_non_bg(self) -> int:
        """
        Compute n'_non_BG ∈ {0, 1} (Proposition (iii) of §5).

        = 1 iff simultaneously:
          (a) no remove is possible (no accessible target-group item on top),
          (b) no empty stack exists, and
          (c) min{gtop_BP(s)} > max{gmin(s)}
              where gtop_BP(s) = top group if badly placed, else n+1.
        """
        # (a) no remove possible
        if any(stk and stk[-1] == self.next_group for stk in self.stacks):
            return 0
        # (b) no empty stack
        if any(not stk for stk in self.stacks):
            return 0
        # (c)
        max_gmin = max((min(stk) for stk in self.stacks if stk), default=0)
        min_gtop_bp = self.n + 1           # sentinel ≡ n+1
        for s in range(self.S):
            stk = self.stacks[s]
            if stk and self.is_top_badly_placed(s):
                g = stk[-1]
                if g < min_gtop_bp:
                    min_gtop_bp = g
        return 1 if min_gtop_bp > max_gmin else 0

    def clean_supply(self) -> int:
        """
        Weighted clean supply Sc(L) = Σ_g  g · sc(g)  (§4.1 and sorting in §6.2).

        sc(g) = number of clean supply slots of group g.
        A slot above the top of a *clean* non-empty stack (all items well-placed)
        whose top has group g counts as one clean supply slot of group g.
        Each empty stack contributes H clean supply slots of group G.
        """
        sc: List[int] = [0] * (self.G + 2)
        for stk in self.stacks:
            if not stk:
                sc[self.G] += self.H
            elif self._stack_is_clean(stk):
                g_top  = stk[-1]
                avail  = self.H - len(stk)
                if avail > 0 and g_top <= self.G:
                    sc[g_top] += avail
        return sum(g * sc[g] for g in range(1, self.G + 1))

    def _stack_is_clean(self, stk: List[int]) -> bool:
        """True iff every item in stk is well placed (no item blocks one below)."""
        if len(stk) < 2:
            return True
        running_min = stk[0]
        for i in range(1, len(stk)):
            if running_min < stk[i]:
                return False
            if stk[i] < running_min:
                running_min = stk[i]
        return True

    # ------------------------------------------------------------------ #
    # Mutation                                                               #
    # ------------------------------------------------------------------ #

    def apply_remove(self, s: int) -> None:
        """
        Remove the topmost item of stack s.
        The item must belong to next_group (caller's responsibility).
        Updates n and next_group.
        """
        self.stacks[s].pop()
        self.n -= 1
        if self.n == 0:
            self.next_group = self.G + 1
            return
        if not any(g == self.next_group for st in self.stacks for g in st):
            self.next_group = min(g for st in self.stacks for g in st)

    def apply_relocation(self, d: int, r: int) -> None:
        """Move top of stack d to top of stack r (no validity check)."""
        g = self.stacks[d].pop()
        self.stacks[r].append(g)

    def do_auto_removes(self) -> int:
        """
        Execute all possible removes (items whose group == next_group on top)
        until no more are possible.  Returns the number of items removed.
        """
        count   = 0
        changed = True
        while changed and not self.is_empty():
            changed = False
            for s in range(self.S):
                stk = self.stacks[s]
                while stk and stk[-1] == self.next_group:
                    self.apply_remove(s)
                    count  += 1
                    changed = True
                    if self.is_empty():
                        return count
        return count
