"""Small optimality checks for the component-based IDA*-R."""

from __future__ import annotations

import random
from types import SimpleNamespace

from algorithms.common.tree.components import (
    LB1,
    LB2,
    LB3,
    PR1,
    PR4,
    PRPlus,
    PRRollout,
    RestrictedCRPSpace,
    SingleMove,
)
from algorithms.common.tree.strategies import IDAStar
from core.container import Container
from core.solver import SolveContext
from core.yard import Yard


class _StopEvent:
    def is_set(self) -> bool:
        return False


class _Problem:
    def __init__(self, stacks, max_tiers):
        self.source = [list(stack) for stack in stacks]
        self.config = SimpleNamespace(
            num_bays=len(stacks),
            num_rows=1,
            max_tiers=max_tiers,
        )
        self.yard = Yard(len(stacks), 1, max_tiers)
        self.containers = []

    def reset(self, options=None):
        del options
        self.yard.clear()
        count = sum(len(stack) for stack in self.source)
        by_id = {
            priority: Container(
                id=priority,
                priority=priority,
            )
            for priority in range(1, count + 1)
        }
        self.containers = list(by_id.values())

        for bay, values in enumerate(self.source, 1):
            stack = self.yard.stacks[(bay, 1)]
            for value in values:
                stack.push(by_id[value])


def _random_instance(width, height, count, seed):
    values = list(range(1, count + 1))
    random.Random(seed).shuffle(values)
    stacks = [[] for _ in range(width)]
    for index, value in enumerate(values):
        stacks[index % width].append(value)
    return stacks


def _brute_force_optimal(stacks, max_tiers):
    best = [10 ** 9]

    def reduce_state(state):
        current = [tuple(stack) for stack in state]
        while True:
            remaining = [
                value
                for stack in current
                for value in stack
            ]
            if not remaining:
                return tuple(current)

            target = min(remaining)
            source = next(
                index
                for index, stack in enumerate(current)
                if target in stack
            )
            if current[source][-1] != target:
                return tuple(current)
            current[source] = current[source][:-1]

    def confirmed(state):
        total = 0
        for stack in state:
            minimum = None
            for value in stack:
                if minimum is not None and value > minimum:
                    total += 1
                minimum = value if minimum is None else min(minimum, value)
        return total

    def visit(state, cost):
        state = reduce_state(state)
        if not any(state):
            best[0] = min(best[0], cost)
            return
        if cost + confirmed(state) >= best[0]:
            return

        target = min(value for stack in state for value in stack)
        source = next(
            index
            for index, stack in enumerate(state)
            if target in stack
        )
        for destination, stack in enumerate(state):
            if destination == source or len(stack) >= max_tiers:
                continue
            child = list(state)
            blocker = child[source][-1]
            child[source] = child[source][:-1]
            child[destination] = child[destination] + (blocker,)
            visit(tuple(child), cost + 1)

    visit(tuple(tuple(stack) for stack in stacks), 0)
    return best[0]


def _run(stacks, max_tiers, bound_type, probe_mode):
    space = RestrictedCRPSpace()
    search = IDAStar(
        state_space=space,
        branching=SingleMove(space),
        lower_bound=bound_type(space),
        upper_bound=PRPlus(space),
        frontier_probe=PRRollout(space, probe_mode),
    )
    return search.solve(
        _Problem(stacks, max_tiers),
        SolveContext(_StopEvent(), 10.0),
    )


def check_instance(width, height, count, seed):
    stacks = _random_instance(width, height, count, seed)
    optimal = _brute_force_optimal(stacks, height)

    problem = _Problem(stacks, height)
    space = RestrictedCRPSpace()
    root = space.initial(problem).state
    bounds = [
        bound_type(space)(problem, root)
        for bound_type in (LB1, LB2, LB3)
    ]
    dominance_ok = bounds[0] <= bounds[1] <= bounds[2] <= optimal

    results = [
        _run(stacks, height, bound_type, probe_mode)
        for bound_type in (LB1, LB2, LB3)
        for probe_mode in (PR1, PR4)
    ]
    all_match = all(
        result.objective == optimal
        and result.status == "optimal"
        and result.plan is not None
        and result.plan.num_relocations() == optimal
        and result.plan.num_retrievals() == count
        for result in results
    )

    ok = dominance_ok and all_match
    label = "OK" if ok else "FAIL"
    print(
        f"[{label}] W={width} H={height} N={count} seed={seed} "
        f"optimal={optimal} bounds={bounds}"
    )
    return ok


def main():
    all_ok = True
    cases = [
        (3, 3, 6),
        (3, 4, 8),
        (4, 3, 8),
        (4, 4, 10),
        (3, 5, 10),
    ]
    for width, height, count in cases:
        for seed in range(6):
            all_ok = (
                check_instance(width, height, count, seed)
                and all_ok
            )
    print("\nALL OK" if all_ok else "\nSOME FAILURES")


if __name__ == "__main__":
    main()
