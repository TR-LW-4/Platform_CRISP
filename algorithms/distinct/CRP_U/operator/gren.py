"""
Gre-N relocation rule for CRP-U
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

from typing import Optional, Tuple

from .grec import candidates


def choose_move(stacks, max_tiers) -> Optional[Tuple[int, int]]:
    options = candidates(stacks, max_tiers, extended=False)
    if not options:
        return None
    source, destination, _ = min(options, key=lambda option: option[2])
    return source, destination
