"""
One ant retrieves the bay. Each blocker is placed by the pheromone transition rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .scoring import compute_lb, dif
from .pheromone import Solution


@dataclass(frozen=True)
class Start:
    stacks: List[List[int]]
    n_total: int
    max_tiers: int
    loc: List[Optional[int]]
    smin: List[int]


def start(stacks: List[List[int]], max_tiers: int) -> Start:
    n_total = sum(len(stack) for stack in stacks)
    loc: List[Optional[int]] = [None] * (n_total + 1)
    for index, stack in enumerate(stacks):
        for container in stack:
            loc[container] = index
    smin = [min(stack) if stack else n_total + 1 for stack in stacks]
    return Start(stacks, n_total, max_tiers, loc, smin)


def build_solution(
    start: Start,
    pheromone: np.ndarray,
    q0: float,
    max_moves: int,
    best_cost: float,
) -> Tuple[Solution, bool, float]:
    n_total = start.n_total
    max_tiers = start.max_tiers
    n_stacks = len(start.stacks)
    stacks = [list(stack) for stack in start.stacks]
    moved = [0] * (n_total + 1)
    solution: Solution = []
    valid = True
    loc = list(start.loc)
    smin = list(start.smin)
    cost = 0.0

    for target in range(1, n_total + 1):
        if not valid:
            break
        src = loc[target]
        if src is None:
            continue

        while valid and stacks[src] and stacks[src][-1] != target:
            blocking = compute_lb({i: stack for i, stack in enumerate(stacks)})
            if cost + blocking >= best_cost:
                valid = False
                break
            container = stacks[src][-1]
            count = min(moved[container], max_moves)
            best_g = -1.0
            best_dst = -1
            best_d = 0
            weights: List[float] = []
            destinations: List[int] = []
            labels: List[int] = []

            for dst in range(n_stacks):
                if dst == src or len(stacks[dst]) >= max_tiers:
                    continue
                earliest = smin[dst]
                label = earliest if earliest <= n_total else n_total + dst + 1
                attractiveness = 1.0 / (1.0 + dif(container, earliest, n_total))
                score = attractiveness * float(pheromone[container - 1, label - 1, count, target - 1])
                weights.append(score)
                destinations.append(dst)
                labels.append(label)
                if score > best_g:
                    best_g = score
                    best_dst = dst
                    best_d = label

            if not destinations:
                valid = False
                break

            if np.random.uniform() < q0 or len(destinations) == 1:
                dst = best_dst
                label = best_d
            else:
                total = float(sum(weights))
                probabilities = (
                    np.asarray(weights) / total
                    if total > 0.0
                    else np.ones(len(destinations)) / len(destinations)
                )
                picked = int(np.random.choice(len(destinations), p=probabilities))
                dst = destinations[picked]
                label = labels[picked]

            solution.append((container, label, count, target))
            stacks[src].pop()
            if smin[src] == container:
                smin[src] = min(stacks[src]) if stacks[src] else n_total + 1
            stacks[dst].append(container)
            if container < smin[dst]:
                smin[dst] = container
            loc[container] = dst
            moved[container] += 1
            cost += 1.0

        if valid and stacks[src] and stacks[src][-1] == target:
            stacks[src].pop()
            if smin[src] == target:
                smin[src] = min(stacks[src]) if stacks[src] else n_total + 1
            loc[target] = None

    return solution, valid, cost


# def decode(solution: Solution, stacks: List[List[int]], n_total: int) -> List[Tuple[int, Optional[int]]]:
#     loc = {container: index for index, stack in enumerate(stacks) for container in stack}
#     by_target = {}
#     for record in solution:
#         by_target.setdefault(record[3], []).append(record)

#     moves: List[Tuple[int, Optional[int]]] = []
#     for target in range(1, n_total + 1):
#         for container, label, _count, _target in by_target.get(target, []):
#             dst = loc[label] if label <= n_total else label - n_total - 1
#             moves.append((loc[container], dst))
#             loc[container] = dst
#         moves.append((loc[target], None))
#         loc[target] = None
#     return moves