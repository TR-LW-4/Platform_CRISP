"""
Move-sequence correction for JovanovicTubaVoss2015MultiHeuristic.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
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
