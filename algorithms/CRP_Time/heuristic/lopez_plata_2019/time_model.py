"""
Crane time f2 / f2vert of Voß & Schwarze (2019) as CRISP defines it (core/objectives.py,
movement_f2 and movement_f2_vertical), as a cost model for the López-Plata search (§4 of
docs/validation/lopez_plata_2019.md). Stacks are numbered 1..n_S, the truck is at 0.

Both are sums of per-move terms that do not depend on the crane position. Every term is an
integer in units of 1/scale s, so g, the incremental h and all ties are exact; seconds()
converts back for reporting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Tuple


@dataclass(frozen=True)
class TimeModel:
    """n_T = max_tiers (stack capacity; h_max = n_T + 1). ts, tpp, tr = tr0 + tr1 and
    tr_hout = tr * hout are scaled by scale."""

    n_T: int
    vertical: bool
    ts: int
    tpp: int
    tr: int
    tr_hout: int
    scale: int

    @classmethod
    def from_spec(cls, spec) -> "TimeModel":
        """From a core.objectives.ObjectiveSpec (ObjectiveSpec.from_config(env.config))."""
        if spec.time_model not in ("f2", "f2_vertical"):
            raise ValueError(f"López-Plata supports the time models f2 and f2_vertical, not {spec.time_model!r}")
        ts, tpp, hout = (Fraction(str(x)) for x in (spec.stack_s_per_stack, spec.pickup_place_s, spec.outside_height))
        tr = Fraction(str(spec.empty_vertical_s_per_tier)) + Fraction(str(spec.loaded_vertical_s_per_tier))
        vertical = spec.time_model == "f2_vertical"
        assert min(ts, tpp, tr, hout) >= 0 and 2 * ts + (2 * tr if vertical else tpp) > 0, \
            "time parameters must be >= 0 and a relocation must take time"
        scale = math.lcm(*(x.denominator for x in (ts, tpp, tr, tr * hout)))
        return cls(int(spec.max_tiers), vertical, int(ts * scale), int(tpp * scale), int(tr * scale),
                   int(tr * hout * scale), scale)

    def seconds(self, value: int) -> float:
        return value / self.scale

    def move(self, crane: Optional[Tuple[int, int]], s: int, pick: int, s2: int, drop: int) -> int:
        """One move from (s, pick) to (s2, drop), s2 = 0 for a retrieval; pick is the tier before the
        move, drop the tier after it. f2: 2 ts d + tpp; f2vert: 2 ts d + tr (2 h_max - pick - drop),
        with hout as the drop height of a retrieval. The crane position plays no role."""
        time = 2 * self.ts * (s if s2 == 0 else abs(s - s2))
        if not self.vertical:
            return time + self.tpp
        if s2 == 0:
            return time + self.tr * (2 * (self.n_T + 1) - pick) - self.tr_hout
        return time + self.tr * (2 * (self.n_T + 1) - pick - drop)

    def h_term(self, s: int, t: int, bad: bool) -> int:
        """Lower bound on the time still spent on the block at (s, t) (§4, L27). Its moves cover at
        least s stacks in total (triangle inequality towards the truck), it is retrieved once, and a
        non-located block is relocated at least once. f2: 2 ts s + tpp (1 + bad). f2vert: a block
        that is never moved is retrieved from tier t; a moved block is picked once from t and placed
        and picked again at most at n_T, which costs at least 2 tiers more: 2 ts s + tr (2 h_max - t
        - hout) + 2 tr bad."""
        if not self.vertical:
            return 2 * self.ts * s + self.tpp * (1 + bad)
        return 2 * self.ts * s + self.tr * (2 * (self.n_T + 1) - t) - self.tr_hout + 2 * self.tr * bad

    def greedy_key(self, s: int, pick: int, d: int, drop: int) -> int:
        """Criterion of the greedy (Sec. 7.1, R1 in crane time): the time of the relocation."""
        return self.move(None, s, pick, d, drop)

    def q_m(self, s: int, n_s: int, d: int, n_d: int) -> int:
        """q_m in crane time: the time of relocating c+ from s to the top of d. The empty return
        trip in q_m has no cost of its own, because f2 and f2vert do not depend on the crane position."""
        return self.move(None, s, n_s, d, n_d + 1)
