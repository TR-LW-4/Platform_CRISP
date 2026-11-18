"""
One Min-Max pass for the CRP-R ant colony.
Each record is (container, label, count, target).

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

from typing import Dict, List, Tuple

from .scoring import dd_star



def greedy(
    stacks_init: List[List[int]],
    n_total: int,
    max_tiers: int,
    max_moves: int,
    choose_destination,
) -> List[Tuple[int, int, int, int]]:
    stacks = [list(stack) for stack in stacks_init]
    moved: Dict[int, int] = {}
    solution: List[Tuple[int, int, int, int]] = []

    for target in range(1, n_total + 1):
        source = next((index for index, stack in enumerate(stacks) if target in stack), -1)
        if source < 0:
            continue

        while stacks[source] and stacks[source][-1] != target:
            container = stacks[source][-1]
            destination = choose_destination(stacks, max_tiers)
            if destination is None:
                raise ValueError(
                    f"no free stack for container {container} above target {target}"
                )
            count = moved.get(container, 0)
            solution.append((
                container,
                dd_star(stacks[destination], n_total, destination + 1),
                min(count, max_moves),
                target,
            ))
            stacks[source].pop()
            stacks[destination].append(container)
            moved[container] = count + 1

        if stacks[source] and stacks[source][-1] == target:
            stacks[source].pop()

    return solution