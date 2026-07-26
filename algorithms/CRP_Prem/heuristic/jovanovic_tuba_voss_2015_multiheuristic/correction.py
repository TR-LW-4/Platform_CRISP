"""
Sec. 4.2: post-processing correction of the generated move sequence
(Eq. 10-11).

Filling, out-of-order block selection, and the deadlock-avoidance mechanism
can all introduce "chain moves": a container relocated twice in a row with
nothing else touching its destination in between. Two local rewrite rules
remove these, applied repeatedly to a fixed point:

    (s1, s2), (s2, s3) -> (s1, s3)      if s1 != s3   (Eq. 10)
    (s1, s2), (s2, s1) -> (removed)                    (Eq. 11)
"""

from __future__ import annotations

from typing import List, Tuple

Move = Tuple[int, int]


def correct_moves(moves: List[Move]) -> List[Move]:
    current = list(moves)
    changed = True
    while changed:
        changed = False
        result: List[Move] = []
        i = 0
        n = len(current)
        while i < n:
            if i + 1 < n:
                s1, s2a = current[i]
                s2b, s3 = current[i + 1]
                if s2a == s2b:
                    if s1 == s3:
                        i += 2
                        changed = True
                        continue
                    result.append((s1, s3))
                    i += 2
                    changed = True
                    continue
            result.append(current[i])
            i += 1
        current = result
    return current
