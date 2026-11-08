from __future__ import annotations

from typing import Dict, List, Tuple

from algorithms.CRP_R.operator.min_max import choose_destination

from .scoring import dd_star


def greedy(
    stacks_init: List[List[int]],
    n_total: int,
    max_tiers: int,
    max_moves: int,
) -> List[Tuple[int, int, int, int]]:
    """Return (container, extended minimum, relocation count, target)."""
    stacks = [list(stack) for stack in stacks_init]
    moved: Dict[int, int] = {}
    solution: List[Tuple[int, int, int, int]] = []

    for target in range(1, n_total + 1):
        src = next((i for i, stack in enumerate(stacks) if target in stack), -1)
        if src < 0:
            continue

        while stacks[src] and stacks[src][-1] != target:
            container = stacks[src][-1]
            dst = choose_destination(stacks, max_tiers)
            if dst is None:
                raise ValueError(
                    f"no free stack for container {container} above target {target}"
                )
            count = moved.get(container, 0)
            solution.append((
                container,
                dd_star(stacks[dst], n_total, dst + 1),
                min(count, max_moves),
                target,
            ))
            stacks[src].pop()
            stacks[dst].append(container)
            moved[container] = count + 1

        if stacks[src] and stacks[src][-1] == target:
            stacks[src].pop()

    return solution