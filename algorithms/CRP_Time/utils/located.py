"""
Whether containers are well located: a container is well located if every
container below it is retrieved later, and non-located (badly placed) otherwise.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from typing import List, Sequence


def min_due(stack: Sequence[int], empty: int) -> int:
    """Lowest retrieval priority in the stack, the most urgent container; empty if there is none."""
    return min(stack) if stack else empty


def non_located(stack: Sequence[int]) -> List[bool]:
    """Per container, bottom to top: True if a container below it is retrieved earlier."""
    flags, lowest = [], math.inf
    for c in stack:
        flags.append(lowest < c)
        lowest = min(lowest, c)
    return flags


def well_placed(c: int, stack: Sequence[int]) -> bool:
    """True if container c would be well located on top of the stack."""
    return not stack or min(stack) > c
