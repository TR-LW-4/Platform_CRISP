"""
Operating cost of López-Plata et al. (2019), eq. 1-3 (p. 437-438).

Stacks are numbered 1..n_S as in the paper; stack 0 is the retrieval point,
whose only slot is tier 1. A move is (s, s') with s' = 0 for a retrieval.
Blocks are identified by their retrieval order r(c) = 1..n_C.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

Move = Tuple[int, int]


@dataclass(frozen=True)
class CostModel:
    """n_T is both the stack capacity and the height term of eq. 1-2 (p. 437).
    Index 1 = crane unloaded, 2 = loaded (p. 437). By default a loaded move costs
    twice as much (Sec. 7.7, p. 449; Table 5, p. 450); the numbers printed at
    Fig. 2 (p. 438) and on p. 445 are the other way round (O01)."""

    n_T: int
    epsilon: float = 2.0
    eta1: float = 1.0
    gamma1: float = 1.0
    eta2: float = 2.0
    gamma2: float = 2.0

    def __post_init__(self):
        assert min(self.eta1, self.gamma1, self.eta2, self.gamma2) > 0, "the weights of eq. 1-2 must be > 0"

    def _trip(self, s1: int, t1: int, s2: int, t2: int, eta: float, gamma: float) -> float:
        # Rise to n_T + epsilon, travel |s1 - s2| stacks, lower to t2.
        top = self.n_T + self.epsilon
        return (top - t1) * eta + abs(s1 - s2) * gamma + (top - t2) * eta

    def m(self, s1: int, t1: int, s2: int, t2: int) -> float:
        """Eq. 1: unloaded crane from (s1, t1) to (s2, t2)."""
        return self._trip(s1, t1, s2, t2, self.eta1, self.gamma1)

    def m_loaded(self, s1: int, t1: int, s2: int, t2: int) -> float:
        """Eq. 2 (m'): loaded crane from (s1, t1) to (s2, t2)."""
        return self._trip(s1, t1, s2, t2, self.eta2, self.gamma2)

    def move(self, crane: Tuple[int, int], s: int, pick: int, s2: int, drop: int) -> float:
        """Term of eq. 3 for one move: the empty trip from the crane to the pickup plus the loaded trip."""
        return self.m(*crane, s, pick) + self.m_loaded(s, pick, s2, drop)

    def h_term(self, s: int, t: int, bad: bool) -> float:
        """Term of the block at (s, t) in h (Sec. 5.1, p. 442).
        EstRet (eq. 21): descend from n_T + epsilon to c, carry c to the retrieval point.
        EstRel (eq. 22), if c is non-located: EstRet plus one relocation to the top tier n_T of the
        adjacent stack towards the retrieval point, s(c) - 1, or s(c) + 1 if s(c) = 1."""
        descend = self.m(s, self.n_T + self.epsilon, s, t)
        term = descend + self.m_loaded(s, t, 0, 1)
        if bad:
            term += descend + self.m_loaded(s, t, s + 1 if s == 1 else s - 1, self.n_T)
        return term

    def greedy_key(self, s: int, pick: int, d: int, drop: int) -> float:
        """Criterion of the greedy (Sec. 7.1, R1): m' of the relocation."""
        return self.m_loaded(s, pick, d, drop)

    def q_m(self, s: int, n_s: int, d: int, n_d: int) -> float:
        """Eq. 27: carry c+ from s to the new top of d, then return empty to the new top of s."""
        return self.m_loaded(s, n_s, d, n_d + 1) + self.m(d, n_d + 1, s, n_s - 1)


class Bay:
    """Configuration and crane position under the restricted rule (p. 437). As a search
    node it also holds phi, the moves from the root, and g = f(phi) (p. 441)."""

    __slots__ = ("stacks", "cost", "crane", "target", "phi", "g")

    def __init__(self, stacks: Sequence[Sequence[int]], cost: CostModel):
        self.stacks: List[List[int]] = [list(s) for s in stacks]  # bottom -> top
        if any(len(s) > cost.n_T for s in self.stacks):
            raise ValueError(f"a stack is higher than n_T = {cost.n_T}")
        self.cost = cost
        self.crane: Tuple[int, int] = (0, 1)  # dummy move (0, 0): start at the retrieval point
        self.target = min((c for s in self.stacks for c in s), default=None)  # c*
        self.phi: List[Move] = []
        self.g = 0.0

    def copy(self) -> "Bay":
        other = Bay.__new__(Bay)
        other.stacks = [list(s) for s in self.stacks]
        other.cost, other.crane, other.target = self.cost, self.crane, self.target
        other.phi, other.g = list(self.phi), self.g
        return other

    def target_stack(self) -> int:
        """s(c*), 1-based."""
        return next(s for s, st in enumerate(self.stacks, 1) if self.target in st)

    def apply(self, move: Move) -> float:
        """Carry out one move; return its term of eq. 3: the empty trip from the
        last drop to the pickup plus the loaded trip to the drop."""
        s, s2 = move
        self._check(s, s2)
        pick = len(self.stacks[s - 1])  # tier before the move
        block = self.stacks[s - 1].pop()
        if s2 == 0:
            drop = 1
            self.target = min((c for st in self.stacks for c in st), default=None)
        else:
            self.stacks[s2 - 1].append(block)
            drop = len(self.stacks[s2 - 1])  # tier after the move
        cost = self.cost.move(self.crane, s, pick, s2, drop)
        self.crane = (s2, drop)
        self.phi.append(move)
        self.g += cost
        return cost

    def undo(self, move: Move, crane: Tuple[int, int], target: Optional[int], g: float) -> None:
        """Reverse apply(move); crane, target and g are the values from before the move."""
        s, s2 = move
        block = target if s2 == 0 else self.stacks[s2 - 1].pop()
        self.stacks[s - 1].append(block)
        self.crane, self.target, self.g = crane, target, g
        self.phi.pop()

    def _check(self, s: int, s2: int) -> None:
        n_S = len(self.stacks)
        if not 1 <= s <= n_S or not 0 <= s2 <= n_S:
            raise ValueError(f"move {(s, s2)}: stack outside 1..{n_S}")
        stack = self.stacks[s - 1]
        if not stack:
            raise ValueError(f"move {(s, s2)}: stack {s} is empty")
        if s2 == 0:
            if stack[-1] != self.target:
                raise ValueError(f"move {(s, s2)}: block {stack[-1]} is not the next to retrieve ({self.target})")
            return
        # Only c+, the block above c*, may be relocated, and not within its own stack.
        if self.target not in stack:
            raise ValueError(f"move {(s, s2)}: stack {s} does not hold c* = {self.target}")
        if stack[-1] == self.target:
            raise ValueError(f"move {(s, s2)}: c* = {self.target} is on top and must be retrieved")
        if s2 == s:
            raise ValueError(f"move {(s, s2)}: relocation within stack {s}")
        if len(self.stacks[s2 - 1]) >= self.cost.n_T:
            raise ValueError(f"move {(s, s2)}: stack {s2} is full")


def f(stacks: Sequence[Sequence[int]], phi: Iterable[Move], cost: CostModel) -> float:
    """Eq. 3 for a (partial) solution phi, starting from the initial configuration."""
    bay = Bay(stacks, cost)
    return sum(bay.apply(move) for move in phi)
