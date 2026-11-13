"""
Crane move time of Forster & Bortfeldt (Oper. Res. Proc. 2011, p. 258) for a
block of bays: per move b*t_b + s*t_s + ad*t_ad + t_pp, with b and s the bays
and stacks the crane crosses from its last position to the starting stack and
on to the target stack, and ad = 1 if b >= 1. Parameters of footnote 1, p. 261
(Lee & Lee 2010). An evaluator only: the search does not use it. Decisions
P4-P8: docs/validation/forster_bortfeldt_2011_multibay.md §2.

A position is (bay, row), bays and rows from 1 inside the block; the vehicle
for removed containers stands at TRUCK and the crane starts at START (P6, P7).

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite
F. Forster, A. Bortfeldt, "A tree search heuristic for the container
retrieval problem", Operations Research Proceedings 2011, 257-262 (2012).
--------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

Position = Tuple[int, int]

TRUCK: Position = (0, 0)
START: Position = TRUCK


@dataclass(frozen=True)
class TimeParams:
    """Footnote 1, p. 261: t_b 3.5 s, t_s 1.2 s, t_ad 40 s, t_pp 30 s."""
    tb: float = 3.5
    ts: float = 1.2
    tad: float = 40.0
    tpp: float = 30.0


def move_time(last: Position, src: Position, dst: Position, p: TimeParams = TimeParams()) -> float:
    """p. 258: b and s summed over last -> src -> dst (P5); one ad per move (P4)."""
    b = abs(last[0] - src[0]) + abs(src[0] - dst[0])
    s = abs(last[1] - src[1]) + abs(src[1] - dst[1])
    return b * p.tb + s * p.ts + (p.tad if b >= 1 else 0.0) + p.tpp


def plan_time(moves: Iterable[Tuple[Position, Optional[Position]]], p: TimeParams = TimeParams(),
              start: Position = START, truck: Position = TRUCK) -> float:
    """Total time of moves (src, dst), dst None for a remove; the crane ends each move at dst (or truck)."""
    last, total = start, 0.0
    for src, dst in moves:
        dst = truck if dst is None else dst
        total += move_time(last, src, dst, p)
        last = dst
    return total
