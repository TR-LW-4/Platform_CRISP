"""
Heuristics shared by SmSEQ, RakeSearch, and Pilot: the SM-1/SM-2 policy,
SmSEQ, and the condensation passes.

------------------------------- Reference --------------------------------
F. Tricoire, J. Scagnetti, A. Beham,
"New insights on the block relocation problem",
Computers & Operations Research 89 (2018) 127–139.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from algorithms.distinct.CRP_U.operator import sm1, sm2


def policy(state, choose):
    while True:
        state.retrieve_all()
        if state.empty():
            return state
        source, destination = choose(state.stacks, state.max_tiers)
        state.relocate(source, destination)


def smseq(state, variant, condensation):
    choose = sm1.choose_move if variant == 1 else sm2.choose_move
    while True:
        state.retrieve_all()
        if state.empty():
            break
        home = state.where[state.next]
        if len(_decreasing_sequence(state.stacks[home], state.next)) < 2:
            source, destination = choose(state.stacks, state.max_tiers)
            state.relocate(source, destination)
        else:
            _make_room(state, home)
            state.relocate(home, _la_destination(state, home))
    if condensation == "tricoire":
        state.ops, state.relocations = _condense(_tricoire_pass, state)
    elif condensation == "jin":
        state.ops, state.relocations = _condense(_jin_pass, state)
    return state


def _decreasing_sequence(stack, target):
    index = len(stack) - 1
    while stack[index - 1] != target and stack[index - 1] <= stack[index]:
        index -= 1
    return stack[index:][::-1]


def _must_move(stack, index):
    return index > 0 and stack[index] > min(stack[:index])


def _la_destination(state, source):
    item = state.stacks[source][-1]
    lows = state.lows()
    best = None
    for index, stack in enumerate(state.stacks):
        if index == source or len(stack) == state.max_tiers:
            continue
        if best is None:
            best = index
        elif lows[index] > item:
            if lows[best] < item or lows[index] < lows[best]:
                best = index
        elif lows[index] < item and lows[best] < item and lows[index] > lows[best]:
            best = index
    return best


def _make_room(state, home):
    sequence = _decreasing_sequence(state.stacks[home], state.next)
    lows = state.lows()
    best_count, best_cost, best_stack = -1, state.max_tiers + 1, -1
    for stack in range(len(state.stacks)):
        if stack != home:
            count, cost = _fit(state, lows, sequence, stack, home)
            if cost < best_cost:
                best_count, best_cost, best_stack = count, cost, stack
    for _ in range(best_count):
        state.relocate(best_stack, _la_destination(state, best_stack))


def _fit(state, lows, sequence, index, home):
    stacks, max_tiers = state.stacks, state.max_tiers
    stack = stacks[index]
    free = sum(max_tiers - len(other) for t, other in enumerate(stacks) if t != index and t != home)
    position = len(stack)
    while (position + len(sequence) > max_tiers
           or (position > 0 and sequence[-1] > min(stack[:position])
               and len(stack) - position < free)):
        position -= 1
    shelter = max((lows[t] for t in range(len(stacks))
                   if t != index and lows[t] != state.next and len(stacks[t]) < max_tiers),
                  default=0)
    cost = 0
    for offset in range(position, len(stack)):
        item = stack[offset]
        cost += 0 if _must_move(stack, offset) else 1
        if shelter < item:
            cost += 1
    # The original code takes the minimum below position - 1, not position.
    local = state.n + 1 if position == 0 else min(stack[:max(position - 1, 1)])
    for item in sequence[:-1]:
        if item <= local:
            break
        cost += 1
    if sequence[-1] > local:
        cost += 1
    return len(stack) - position, cost


def _condense(single_pass, state):
    ops, relocations = list(state.ops), state.relocations
    while True:
        before = relocations
        relocations = single_pass(ops, relocations, state.heights, state.max_tiers)
        if relocations == before:
            return ops, relocations


# Both passes bound their indices by the relocation count, not by len(ops),
# exactly as the original code does.
def _tricoire_pass(ops, relocations, heights, max_tiers):
    rows = [list(heights)]
    for source, destination in ops:
        row = list(rows[-1])
        row[source] -= 1
        if destination != source:
            row[destination] += 1
        rows.append(row)
    first = 0
    while first < relocations - 1:
        s1, s2 = ops[first]
        if s1 == s2:
            first += 1
            continue
        second = first + 1
        while second < relocations:
            if ops[second][0] == ops[second][1] or ops[second][0] != s2:
                second += 1
                continue
            s3 = ops[second][1]
            if s1 != s3 and rows[first][s3] == max_tiers:
                second += 1
                continue
            ok, above_c, above_s3 = True, 0, 0
            for i in range(first + 1, second):
                source, destination = ops[i]
                if above_c == 0 and source == s2:
                    ok = False
                    break
                if source != s2 and destination == s2:
                    above_c += 1
                if source == s2:
                    above_c -= 1
                if destination == s3:
                    if source == s3:
                        ok = False
                        break
                    above_s3 += 1
                    if above_s3 + rows[i][s3] >= max_tiers - 1:
                        ok = False
                        break
                if source == s3:
                    above_s3 -= 1
                    if above_s3 < 0:
                        ok = False
                        break
            if not ok or above_s3 != 0 or above_c != 0:
                second += 1
                continue
            for i in range(first + 1, second):
                rows[i][s2] -= 1
                rows[i][s3] += 1
            del rows[second]
            del ops[second]
            relocations -= 1
            ops[first] = (s1, s3)
            s2 = s3
            if s1 == s3:
                del rows[first]
                del ops[first]
                relocations -= 1
                while first < len(ops) and ops[first][0] == ops[first][1]:
                    first += 1
                if first == len(ops):
                    return relocations
                s1, s2 = ops[first]
                second = first + 1
        first += 1
    return relocations


def _jin_pass(ops, relocations, heights, max_tiers):
    first = 0
    while first < relocations - 1:
        while first < len(ops) and ops[first][0] == ops[first][1]:
            first += 1
        if first == len(ops):
            return relocations
        s1, s2 = ops[first]
        second = first + 1
        while second < relocations:
            if ops[second][0] == ops[second][1] or ops[second][0] != s2:
                second += 1
                continue
            s3 = ops[second][1]
            ok, above_c = True, 0
            for i in range(first + 1, second):
                source, destination = ops[i]
                if above_c == 0 and source == s2:
                    ok = False
                    break
                if source != s2 and destination == s2:
                    above_c += 1
                if source == s2:
                    above_c -= 1
                if source == s3 or destination == s3:
                    ok = False
                    break
            if not ok or above_c != 0:
                second += 1
                continue
            del ops[second]
            relocations -= 1
            ops[first] = (s1, s3)
            s2 = s3
            if s1 == s3:
                del ops[first]
                relocations -= 1
                while first < len(ops) and ops[first][0] == ops[first][1]:
                    first += 1
                if first == len(ops):
                    return relocations
                s1, s2 = ops[first]
                second = first + 1
        first += 1
    return relocations
