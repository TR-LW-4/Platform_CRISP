"""
Operating cost of López-Plata et al. (2019), p. 437-438: the paper's own cost model,
used only for the validation in the paper's setting. The search runs on time_model.TimeModel.

Stacks are numbered 1..n_S as in the paper; stack 0 is the retrieval point,
whose only slot is tier 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class CostModel:
    """n_T is both the stack capacity and the height term of the trip costs m and m' (p. 437).
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
        assert min(self.eta1, self.gamma1, self.eta2, self.gamma2) > 0, "the trip weights eta and gamma must be > 0"

    def _trip(self, s1: int, t1: int, s2: int, t2: int, eta: float, gamma: float) -> float:
        # Rise to n_T + epsilon, travel |s1 - s2| stacks, lower to t2.
        top = self.n_T + self.epsilon
        return (top - t1) * eta + abs(s1 - s2) * gamma + (top - t2) * eta

    def m(self, s1: int, t1: int, s2: int, t2: int) -> float:
        """m: trip of the unloaded crane from (s1, t1) to (s2, t2)."""
        return self._trip(s1, t1, s2, t2, self.eta1, self.gamma1)

    def m_loaded(self, s1: int, t1: int, s2: int, t2: int) -> float:
        """m': trip of the loaded crane from (s1, t1) to (s2, t2)."""
        return self._trip(s1, t1, s2, t2, self.eta2, self.gamma2)

    def move(self, crane: Tuple[int, int], s: int, pick: int, s2: int, drop: int) -> float:
        """Cost of one move: the empty trip from the crane to the pickup plus the loaded trip."""
        return self.m(*crane, s, pick) + self.m_loaded(s, pick, s2, drop)

    def h_term(self, s: int, t: int, bad: bool) -> float:
        """Term of the block at (s, t) in h (Sec. 5.1, p. 442).
        EstRet: descend from n_T + epsilon to c, carry c to the retrieval point.
        EstRel, if c is non-located: EstRet plus one relocation to the top tier n_T of the
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
        """q_m of step 2: carry c+ from s to the new top of d, then return empty to the new top of s."""
        return self.m_loaded(s, n_s, d, n_d + 1) + self.m(d, n_d + 1, s, n_s - 1)
