"""Online retrieval with container groups: group order is known, ids inside a group stay fixed."""

from problems.MOVE import Move
from problems.PROBLEM import BaseProblem


def targets(remaining, data):
    for box_id in data:
        if box_id in remaining:
            return (box_id,)
    return ()


def replay(problem, relocations):
    """Turn relocations [(source, destination), ...] into the full move sequence.
    A container is retrieved as soon as it is the next target and on top."""
    state = [list(stack) for stack in problem.stacks]
    moves = []

    def retrieve():
        while True:
            wanted = problem.targets({c for stack in state for c in stack})
            top = next((s for s, stack in enumerate(state) if stack and stack[-1] in wanted), None)
            if top is None:
                return
            moves.append(Move(state[top].pop(), (top, len(state[top]) + 1), None))

    retrieve()
    for source, destination in relocations:
        if source == destination or not state[source] or len(state[destination]) >= problem.max_tiers:
            raise ValueError(f"illegal relocation {(source, destination)}")
        container = state[source].pop()
        state[destination].append(container)
        moves.append(Move(container, (source, len(state[source]) + 1),
                        (destination, len(state[destination]))))
        retrieve()
    if any(state):
        raise ValueError("relocations do not empty the bay")
    return moves


def group_map(order, group_size, num_groups):
    """Cut the true retrieval sequence into contiguous groups.

    ``group_size`` > 0 cuts every that many containers. Otherwise the sequence
    is split into ``num_groups`` consecutive blocks.
    """
    order = list(order)
    size = int(group_size or 0)
    if size > 0:
        return {container: index // size for index, container in enumerate(order)}
    count = int(num_groups or 0)
    count = min(count, len(order)) if count > 0 else 1
    base, extra = divmod(len(order), count)
    mapping = {}
    index = 0
    for group in range(count):
        for _ in range(base + (1 if group < extra else 0)):
            mapping[order[index]] = group
            index += 1
    return mapping


def group_view(container, next_target, order, mapping, reveal):
    """Return the comparable rank. A smaller rank leaves sooner.

    A hidden container uses the last position of its group, so containers in
    the same hidden group tie.
    """
    position = {box: index for index, box in enumerate(order)}
    group = mapping[container]
    mode = str(reveal)
    if mode == "next":
        known = container == next_target
    elif mode == "current":
        known = group == mapping[next_target]
    else:
        raise ValueError(f"unknown reveal {reveal}")
    if known:
        return position[container]
    return max(position[box] for box, owner in mapping.items() if owner == group)


class CRP_OG_R(BaseProblem):
    """Container groups. ``group_size`` > 0 wins; otherwise use ``num_groups``.
    ``reveal`` is ``next`` (only the next id) or ``current`` (the open group)."""

    name = "CRP-OG-R"
    group_size = 0
    num_groups = 2
    reveal = "current"

    def targets(self, remaining):
        return targets(remaining, self.retrieval_order)

    def score(self, container, target):
        mapping = group_map(self.retrieval_order, self.group_size, self.num_groups)
        return group_view(container, target, self.retrieval_order, mapping, self.reveal)

    def replay(self, relocations):
        return replay(self, relocations)


class CRP_OG_U(BaseProblem):
    """Container groups. ``group_size`` > 0 wins; otherwise use ``num_groups``.
    ``reveal`` is ``next`` (only the next id) or ``current`` (the open group)."""

    name = "CRP-OG-U"
    group_size = 0
    num_groups = 2
    reveal = "current"

    def targets(self, remaining):
        return targets(remaining, self.retrieval_order)

    def score(self, container, target):
        mapping = group_map(self.retrieval_order, self.group_size, self.num_groups)
        return group_view(container, target, self.retrieval_order, mapping, self.reveal)

    def replay(self, relocations):
        return replay(self, relocations)
