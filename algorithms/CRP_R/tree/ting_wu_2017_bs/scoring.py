"""SDH and VRH evaluators from Ting and Wu (2017)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from algorithms.common.tree.components import RestrictedCRPSpace

PriorityState = Dict[int, List[int]]


def _stack_min(stack: List[int], n_total: int) -> int:
    return min(stack) if stack else n_total + 1


def _sdh_destination(
    blocker: int,
    source: int,
    stacks: PriorityState,
    n_total: int,
    max_tiers: int,
) -> Optional[int]:
    good: List[Tuple[int, int]] = []
    bad: List[Tuple[int, int]] = []

    for destination, stack in stacks.items():
        if destination == source or len(stack) >= max_tiers:
            continue
        difference = _stack_min(stack, n_total) - blocker
        if difference > 0:
            good.append((difference, destination))
        else:
            bad.append((difference, destination))

    if good:
        return min(good)[1]
    if bad:
        return max(bad)[1]
    return None


def sdh_simulate(
    stacks_in: PriorityState,
    n_total: int,
    max_tiers: int,
) -> int:
    """Return the SDH completion cost."""
    stacks = {
        index: list(stack)
        for index, stack in stacks_in.items()
    }
    relocations = 0

    for target in range(1, n_total + 1):
        source = next(
            (
                index
                for index, stack in stacks.items()
                if target in stack
            ),
            None,
        )
        if source is None:
            continue

        while stacks[source] and stacks[source][-1] != target:
            blocker = stacks[source][-1]
            destination = _sdh_destination(
                blocker,
                source,
                stacks,
                n_total,
                max_tiers,
            )
            if destination is None:
                return n_total * max_tiers
            stacks[source].pop()
            stacks[destination].append(blocker)
            relocations += 1

        if stacks[source] and stacks[source][-1] == target:
            stacks[source].pop()

    return relocations


def _vrh_assign(
    blockers: List[int],
    source: int,
    stacks: PriorityState,
    n_total: int,
    max_tiers: int,
) -> Dict[int, int]:
    assignments: Dict[int, int] = {}
    keys = list(stacks)
    virtual_min = {
        key: _stack_min(stacks[key], n_total)
        for key in keys
    }
    minimum_tier = {key: float("inf") for key in keys}
    assigned: Dict[int, List[Tuple[int, int]]] = {
        key: []
        for key in keys
    }
    heights = {
        key: len(stacks[key])
        for key in keys
    }

    for tier in sorted(
        range(len(blockers)),
        key=lambda index: -blockers[index],
    ):
        blocker = blockers[tier]
        valid = []
        for destination in keys:
            if (
                destination == source
                or heights[destination] >= max_tiers
            ):
                continue
            difference = virtual_min[destination] - blocker
            if difference > 0 and tier < minimum_tier[destination]:
                valid.append((difference, destination))

        if valid:
            _, destination = min(valid)
            assignments[tier] = destination
            virtual_min[destination] = blocker
            minimum_tier[destination] = min(
                minimum_tier[destination],
                tier,
            )
            assigned[destination].append((blocker, tier))
            heights[destination] += 1

    skipped = [
        tier
        for tier in range(len(blockers))
        if tier not in assignments
    ]
    skipped.sort(key=lambda tier: blockers[tier])

    for tier in skipped:
        blocker = blockers[tier]
        best_destination = None
        best_value = float("-inf")

        for destination in keys:
            if (
                destination == source
                or heights[destination] >= max_tiers
            ):
                continue

            uppers = [
                priority
                for priority, assigned_tier in assigned[destination]
                if assigned_tier < tier and priority > blocker
            ]
            lowers = [
                priority
                for priority, assigned_tier in assigned[destination]
                if assigned_tier > tier
            ] + stacks[destination]

            upper_min = min(uppers) if uppers else n_total + 1
            lower_min = min(lowers) if lowers else n_total + 1
            if len(uppers) <= 1:
                value = min(
                    lower_min - blocker,
                    blocker - upper_min,
                )
            else:
                value = -n_total + blocker - upper_min

            if value > best_value:
                best_value = value
                best_destination = destination

        if best_destination is None:
            best_destination = _sdh_destination(
                blocker,
                source,
                stacks,
                n_total,
                max_tiers,
            )

        if best_destination is not None:
            assignments[tier] = best_destination
            assigned[best_destination].append((blocker, tier))
            virtual_min[best_destination] = min(
                virtual_min[best_destination],
                blocker,
            )
            heights[best_destination] += 1

    return assignments


def vrh_simulate(
    stacks_in: PriorityState,
    n_total: int,
    max_tiers: int,
) -> int:
    """Return the VRH completion cost."""
    stacks = {
        index: list(stack)
        for index, stack in stacks_in.items()
    }
    relocations = 0

    for target in range(1, n_total + 1):
        source = next(
            (
                index
                for index, stack in stacks.items()
                if target in stack
            ),
            None,
        )
        if source is None:
            continue

        target_tier = stacks[source].index(target)
        blockers = stacks[source][target_tier + 1:]
        if not blockers:
            stacks[source].pop()
            continue

        assignments = _vrh_assign(
            blockers,
            source,
            stacks,
            n_total,
            max_tiers,
        )
        for tier in range(len(blockers) - 1, -1, -1):
            destination = assignments.get(tier)
            if destination is None:
                destination = _sdh_destination(
                    blockers[tier],
                    source,
                    stacks,
                    n_total,
                    max_tiers,
                )
            if destination is None:
                return n_total * max_tiers
            stacks[source].pop()
            stacks[destination].append(blockers[tier])
            relocations += 1

        if stacks[source] and stacks[source][-1] == target:
            stacks[source].pop()

    return relocations


class TingEvaluator:
    """Adapt VRH or SDH to the shared tree state."""

    def __init__(
        self,
        space: RestrictedCRPSpace,
        mode: str = "VRH",
    ) -> None:
        self.space = space
        self.mode = mode.upper() if mode.upper() in {"VRH", "SDH"} else "VRH"

    def __call__(self, problem, state) -> float:
        del problem
        stacks = {
            index: [
                self.space.priority(container_id)
                for container_id in stack
            ]
            for index, stack in enumerate(state)
        }
        evaluate = vrh_simulate if self.mode == "VRH" else sdh_simulate
        return float(
            evaluate(
                stacks,
                len(self.space.priorities),
                self.space.max_tiers,
            )
        )
