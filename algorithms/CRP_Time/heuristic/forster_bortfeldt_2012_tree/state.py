"""
The layout as the tree search of Forster & Bortfeldt (2012) uses it: immutable
stacks of item groups (bottom to top) with, per stack, the running minimum and
the number of badly placed items, so that a move, the lower bound (Sec. 5,
p. 302), the clean supply (Table 1, p. 302) and the move types (Sec. 4.2,
p. 302) cost O(S) instead of O(n). layout.py holds the plain definitions these
must equal (test_forster_bortfeldt.py). Decisions F1-F19:
docs/validation/forster_bortfeldt_2012.md §2.

A move is (d, r, g): the top item, of group g, of stack d goes to stack r, or
leaves the bay if r is None (a remove).

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite
F. Forster, A. Bortfeldt, "A tree search procedure for the container
relocation problem", Computers & Operations Research 39 (2012) 299-309.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

Move = Tuple[int, Optional[int], int]


class Bay:
    """A layout with maximum stack height H; G is the largest group of the instance (F13)."""

    __slots__ = ("stacks", "mins", "bads", "H", "G", "n", "nb", "_lb")

    def __init__(self, stacks, mins, bads, H: int, G: int, n: int, nb: int):
        self.stacks, self.mins, self.bads = stacks, mins, bads
        self.H, self.G, self.n, self.nb = H, G, n, nb
        self._lb = None

    @classmethod
    def from_stacks(cls, stacks: Sequence[Sequence[int]], H: int, G: Optional[int] = None) -> "Bay":
        st, mn, bd = [], [], []
        for s in stacks:
            m, b, low = [], 0, None
            for g in s:
                b += low is not None and low < g
                low = g if low is None else min(low, g)
                m.append(low)
            st.append(tuple(s))
            mn.append(tuple(m))
            bd.append(b)
        if any(len(s) > H for s in st):
            raise ValueError("a stack is higher than H")
        n = sum(map(len, st))
        G = max((max(s) for s in st if s), default=0) if G is None else G
        return cls(tuple(st), tuple(mn), tuple(bd), H, G, n, sum(bd))

    def low(self) -> int:
        """The currently lowest group."""
        return min(m[-1] for m in self.mins if m)

    def top_bad(self, s: int) -> bool:
        st = self.stacks[s]
        return len(st) > 1 and self.mins[s][-2] < st[-1]

    def removes(self) -> List[Move]:
        """All removes applicable to the layout (p. 302)."""
        if self.n == 0:
            return []
        low = self.low()
        return [(s, None, low) for s, st in enumerate(self.stacks) if st and st[-1] == low]

    def apply(self, move: Move) -> "Bay":
        d, r, _g = move
        st, mn, bd = list(self.stacks), list(self.mins), list(self.bads)
        g, bad_before = st[d][-1], self.top_bad(d)
        st[d], mn[d], bd[d] = st[d][:-1], mn[d][:-1], bd[d] - bad_before
        if r is None:
            return Bay(tuple(st), tuple(mn), tuple(bd), self.H, self.G, self.n - 1, self.nb - bad_before)
        low_r = mn[r][-1] if st[r] else None
        bad_after = low_r is not None and low_r < g
        st[r], mn[r], bd[r] = st[r] + (g,), mn[r] + (g if low_r is None else min(low_r, g),), bd[r] + bad_after
        return Bay(tuple(st), tuple(mn), tuple(bd), self.H, self.G, self.n, self.nb - bad_before + bad_after)

    def lower_bound(self) -> int:
        """n'_m = n'_REM + n'_BG + n'_non-BG of the Proposition (p. 302); g_top as in Table 1 (F2)."""
        if self._lb is None:
            non_bg = 0
            if self.n and all(self.stacks):
                low = self.low()
                if not any(s[-1] == low for s in self.stacks):
                    g_top = min(s[-1] if self.top_bad(i) else self.n + 1 for i, s in enumerate(self.stacks))
                    non_bg = int(g_top > max(m[-1] for m in self.mins))
            self._lb = self.n + self.nb + non_bg
        return self._lb

    def clean_supply(self) -> int:
        """Sc(L) of Table 1 (p. 302): per clean stack (H - height) slots of its top group; H slots of group G per empty stack."""
        return sum(self.H * self.G if not s else (self.H - len(s)) * s[-1] if not b else 0
                   for s, b in zip(self.stacks, self.bads))

    def relocations(self):
        """Every applicable relocation with its type (Sec. 4.2, p. 302), stacks d then r ascending (F8):
        (move, badly placed before, badly placed after, FLG)."""
        low = self.low()
        for d, sd in enumerate(self.stacks):
            if not sd:
                continue
            g, before = sd[-1], self.top_bad(d)
            flg = before and self.mins[d][-2] == low
            for r, sr in enumerate(self.stacks):
                if r != d and len(sr) < self.H:
                    yield (d, r, g), before, bool(sr) and self.mins[r][-1] < g, flg
