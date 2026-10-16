"""
Lower bound of Azari2017CSUM on the movements still needed to empty the bay
(Sec. 4.1, p. 312, from Kim & Hong 2006): 2x + y, with x the bad and y the good
containers. A container is bad if any container below it is retrieved earlier.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Sequence

from algorithms.CRP_Time.utils.located import non_located


def lower_bound(stacks: Sequence[Sequence[int]]) -> int:
    """A bad container needs a relocation and a retrieval, a good one only its retrieval."""
    return sum(2 if bad else 1 for st in stacks for bad in non_located(st))
