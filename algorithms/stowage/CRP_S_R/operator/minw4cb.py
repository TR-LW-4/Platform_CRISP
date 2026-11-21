"""
MinW4CB retrieval and MinMax4CB relocation for a stowage plan.
<2019> <heuristic> <restricted> <single-bay> <CRP-Stow-R>

------------------------------- Reference --------------------------------
R. Jovanović, S. Tanaka, T. Nishi, S. Voß,
"A GRASP approach for solving the Blocks Relocation Problem with Stowage Plan",
Flexible Services and Manufacturing Journal 31 (2019) 702–729.
A chosen target is the highest container whose vessel tier is already due.
Until that container is retrieved, only the top of its yard stack is relocated.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations


_MISSING = 10 ** 6
_ABOVE_WELL = 10_000
_ABOVE_CYCLE = 100


def choose_move(stacks, max_tiers):
    """Return one relocation ``(source, destination)``, or ``None``.

    ``None`` means a yard top is already due and should be retrieved, or the
    yard is empty. The source is a newly chosen target stack.
    """
    if not any(stacks):
        return None
    loaded = _loaded(stacks)
    if any(stack and _due(stack[-1], loaded) for stack in stacks):
        return None
    return _select(stacks, max_tiers)


def relocations(stacks, max_tiers, moved=2, block=1, rng=None):
    """Greedy restricted sequence, then one correction pass.

    ``rng`` breaks ties during construction. Correction stays deterministic,
    and a missing generator keeps the original lowest-index choice.
    """
    state = _normalize(stacks)
    moves, records = _greedy(state, max_tiers, rng=rng)
    return _correct(state, max_tiers, moves, records, moved, block)


def _greedy(stacks, max_tiers, lock=None, block=None, relocated=None, rng=None):
    state = [list(stack) for stack in stacks]
    moves = []
    records = []
    relocated = {} if relocated is None else dict(relocated)
    active = block
    limit = sum(len(stack) for stack in state) * max(max_tiers, 1) * max(len(state), 1) + 5
    while any(state):
        if len(moves) > limit:
            raise ValueError("MinW4CB did not empty the yard")
        if lock is None:
            chosen = _select(state, max_tiers, rng)
            if chosen is None:
                break
            source, destination = chosen
            active = 0 if active is None else active + 1
        else:
            source = lock
            destination = _destination(state, source, max_tiers, rng=rng)
            if destination is None:
                raise ValueError("no stack can accept the relocated container")
        container = state[source][-1]
        well = _well(container, state[destination])
        state[source].pop()
        state[destination].append(container)
        relocated[container] = relocated.get(container, 0) + 1
        moves.append((source, destination))
        records.append({
            "source": source,
            "destination": destination,
            "container": container,
            "well": well,
            "times": relocated[container],
            "block": active,
        })
        before = _count(state)
        state = _normalize(state)
        lock = source if _count(state) == before else None
    if any(state):
        raise ValueError("MinW4CB left containers in the yard")
    return moves, records


def _correct(stacks, max_tiers, moves, records, moved, block):
    best = list(moves)
    best_records = records
    suspicious = _suspicious(records, moved, block)
    index = 0
    while index < len(best):
        if index not in suspicious:
            index += 1
            continue
        record = best_records[index]
        excluded = {record["destination"]}
        improved = False
        while True:
            state = _replay(stacks, best[:index])
            source = record["source"]
            if not state[source] or state[source][-1] != record["container"]:
                break
            destination = _destination(state, source, max_tiers, excluded)
            if destination is None:
                break
            container = state[source][-1]
            well = _well(container, state[destination])
            state[source].pop()
            state[destination].append(container)
            relocated = _times(best_records[:index])
            relocated[container] = relocated.get(container, 0) + 1
            replacement = {
                "source": source,
                "destination": destination,
                "container": container,
                "well": well,
                "times": relocated[container],
                "block": record["block"],
            }
            before = _count(state)
            state = _normalize(state)
            lock = source if _count(state) == before else None
            rest, rest_records = _greedy(
                state, max_tiers, lock, record["block"], relocated,
            )
            candidate = best[:index] + [(source, destination)] + rest
            if len(candidate) < len(best):
                best = candidate
                best_records = best_records[:index] + [replacement] + rest_records
                suspicious = _suspicious(best_records, moved, block)
                index = 0
                improved = True
                break
            excluded.add(destination)
            if not well:
                break
        if not improved:
            index += 1
    return best


def _select(stacks, max_tiers, rng=None):
    loaded = _loaded(stacks)
    well = _well_map(stacks)
    pool = []
    for index, stack in enumerate(stacks):
        target = _top_target(stack, loaded)
        if target is None or target == len(stack) - 1:
            continue
        above_well = above_cycle = above_other = 0
        for position in range(target + 1, len(stack)):
            container = stack[position]
            if not well[container]:
                above_other += 1
            elif _cycle_blocks(stacks, container, index, position, well):
                above_cycle += 1
            else:
                above_well += 1
        score = _ABOVE_WELL * above_well + _ABOVE_CYCLE * above_cycle + above_other
        pool.append((score, index))
    if not pool:
        return None
    source = _pick(pool, rng, 1)
    destination = _destination(stacks, source, max_tiers, rng=rng)
    if destination is None:
        raise ValueError("no stack can accept the relocated container")
    return source, destination


def _destination(stacks, source, max_tiers, excluded=(), rng=None):
    container = stacks[source][-1]
    vessel_stack, vessel_tier = container
    loaded = _loaded(stacks)
    well = _well_map(stacks)
    penalty = _MISSING
    located = []
    elsewhere = []
    for index, stack in enumerate(stacks):
        if index == source or index in excluded or len(stack) >= max_tiers:
            continue
        due = _stack_due(stack, vessel_stack)
        current = _stack_current(stack, loaded)
        if due > vessel_tier:
            cycle = _creates_cycle(stacks, container, index, well)
            located.append((due + penalty * int(cycle), -current, index))
        else:
            elsewhere.append((-due, -current, index))
    pool = located or elsewhere
    if not pool:
        return None
    return _pick(pool, rng, 2)


def _pick(pool, rng, key_len):
    """Lowest key. A generator chooses uniformly among equal score prefixes."""
    ordered = sorted(pool)
    if rng is None:
        return ordered[0][-1]
    best = ordered[0][:key_len]
    tied = [item[-1] for item in ordered if item[:key_len] == best]
    return rng.choice(tied)


def _suspicious(records, moved, block):
    found = set()
    by_container = {}
    by_block = {}
    for index, record in enumerate(records):
        by_container.setdefault(record["container"], []).append(index)
        by_block.setdefault(record["block"], []).append(index)
    for steps in by_container.values():
        if len(steps) >= moved:
            found.add(steps[0])
    for steps in by_block.values():
        if len(steps) < block:
            continue
        for index in steps:
            if records[index]["times"] > 1:
                container = records[index]["container"]
                found.add(max(step for step in steps if records[step]["container"] == container))
    placed = {}
    for index, record in enumerate(records):
        container = record["container"]
        if record["well"]:
            placed[container] = index
        elif container in placed:
            found.add(placed.pop(container))
    return found


def _replay(stacks, moves):
    state = [list(stack) for stack in stacks]
    for source, destination in moves:
        container = state[source].pop()
        state[destination].append(container)
        state = _normalize(state)
    return state


def _normalize(stacks):
    state = [list(stack) for stack in stacks]
    progressed = True
    while progressed:
        progressed = False
        loaded = _loaded(state)
        for stack in state:
            if not stack or not _due(stack[-1], loaded):
                continue
            stack.pop()
            progressed = True
            break
    return state


def _loaded(stacks):
    nxt = {}
    for stack in stacks:
        for vessel_stack, vessel_tier in stack:
            if vessel_stack not in nxt or vessel_tier < nxt[vessel_stack]:
                nxt[vessel_stack] = vessel_tier
    return nxt


def _due(container, loaded):
    vessel_stack, vessel_tier = container
    return vessel_tier == loaded.get(vessel_stack)


def _top_target(stack, loaded):
    for index in range(len(stack) - 1, -1, -1):
        if _due(stack[index], loaded):
            return index
    return None


def _well(container, below):
    vessel_stack, vessel_tier = container
    return not any(
        other_stack == vessel_stack and other_tier < vessel_tier
        for other_stack, other_tier in below
    )


def _well_map(stacks):
    return {
        container: _well(container, stack[:index])
        for stack in stacks
        for index, container in enumerate(stack)
    }


def _stack_due(stack, vessel_stack):
    tiers = [tier for stack_index, tier in stack if stack_index == vessel_stack]
    return min(tiers) if tiers else _MISSING


def _stack_current(stack, loaded):
    if not stack:
        return 0
    return min(tier - loaded.get(vessel_stack, 0) for vessel_stack, tier in stack)


def _cycle_blocks(stacks, container, stack_index, position, well):
    if not well[container]:
        return False
    vessel_stack, vessel_tier = container
    below = stacks[stack_index][:position]
    for lower in below:
        if not well[lower]:
            continue
        lower_stack, lower_tier = lower
        for column in stacks:
            for upper_index, upper in enumerate(column):
                if upper[0] != lower_stack or upper[1] <= lower_tier or not well[upper]:
                    continue
                for earlier in column[:upper_index]:
                    if (
                        earlier[0] == vessel_stack
                        and earlier[1] < vessel_tier
                        and well[earlier]
                    ):
                        return True
    return False


def _creates_cycle(stacks, container, destination, well):
    vessel_stack, vessel_tier = container
    below = stacks[destination]
    if not _well(container, below):
        return False
    for lower in below:
        if not well[lower]:
            continue
        lower_stack, lower_tier = lower
        for column in stacks:
            for upper_index, upper in enumerate(column):
                if upper[0] != lower_stack or upper[1] <= lower_tier or not well[upper]:
                    continue
                for earlier in column[:upper_index]:
                    if (
                        earlier[0] == vessel_stack
                        and earlier[1] < vessel_tier
                        and well[earlier]
                    ):
                        return True
    return False


def _count(stacks):
    return sum(len(stack) for stack in stacks)


def _times(records):
    relocated = {}
    for record in records:
        relocated[record["container"]] = record["times"]
    return relocated
