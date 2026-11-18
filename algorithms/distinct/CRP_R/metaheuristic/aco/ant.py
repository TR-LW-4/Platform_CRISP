"""
One ant retrieves the bay. Each blocker is placed by the pheromone transition rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from algorithms.distinct.CRP_R.LB.lb1 import lb1
from .scoring import dif
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
        source = loc[target]
        if source is None:
            continue

        while valid and stacks[source] and stacks[source][-1] != target:
            lower_bound = lb1(stacks, max_tiers)
            if cost + lower_bound >= best_cost:
                valid = False
                break
            container = stacks[source][-1]
            count = min(moved[container], max_moves)
            best_g = -1.0
            best_destination = -1
            best_label = 0
            weights: List[float] = []
            destinations: List[int] = []
            labels: List[int] = []

            for index in range(n_stacks):
                if index == source or len(stacks[index]) >= max_tiers:
                    continue
                earliest = smin[index]
                label = earliest if earliest <= n_total else n_total + index + 1
                attractiveness = 1.0 / (1.0 + dif(container, earliest, n_total))
                score = attractiveness * float(pheromone[container - 1, label - 1, count, target - 1])
                weights.append(score)
                destinations.append(index)
                labels.append(label)
                if score > best_g:
                    best_g = score
                    best_destination = index
                    best_label = label

            if not destinations:
                valid = False
                break

            if np.random.uniform() < q0 or len(destinations) == 1:
                destination = best_destination
                label = best_label
            else:
                total = float(sum(weights))
                probabilities = (
                    np.asarray(weights) / total
                    if total > 0.0
                    else np.ones(len(destinations)) / len(destinations)
                )
                picked = int(np.random.choice(len(destinations), p=probabilities))
                destination = destinations[picked]
                label = labels[picked]

            solution.append((container, label, count, target))
            stacks[source].pop()
            if smin[source] == container:
                smin[source] = min(stacks[source]) if stacks[source] else n_total + 1
            stacks[destination].append(container)
            if container < smin[destination]:
                smin[destination] = container
            loc[container] = destination
            moved[container] += 1
            cost += 1.0

        if valid and stacks[source] and stacks[source][-1] == target:
            stacks[source].pop()
            if smin[source] == target:
                smin[source] = min(stacks[source]) if stacks[source] else n_total + 1
            loc[target] = None

    return solution, valid, cost