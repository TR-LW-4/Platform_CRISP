"""
Stack label and relocation score for the CRP-R ant colony.

------------------------------- Reference --------------------------------
R. Jovanović, M. Tuba, S. Voß,
"An efficient ant colony optimization algorithm for the blocks relocation
 problem",
European Journal of Operational Research 274 (2019) 78–90.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List


def dd_star(stack: List[int], n_total: int, index: int) -> int:
    if not stack:
        return n_total + index
    return min(stack)


def dif(container: int, earliest: int, n_total: int) -> int:
    if earliest > container:
        return earliest - container
    return 2 * n_total + 1 - earliest