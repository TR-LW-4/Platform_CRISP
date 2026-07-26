"""
Speedy Task Accomplishment Procedure (STAP) -- paper §5.3.

Helper functions implemented paper-faithfully
-----------------------------------------------
* ``alpha`` (Algorithm 3): slack accounting used to filter destinations for
  the "above target" relocations of internal-task Case I1.
* ``relocate`` (Algorithm 4) and ``eval_move`` (Algorithm 5, the four-branch
  stability/messiness destination preference).
* ``interim`` / ``interim_full`` (Algorithm 6).
* ``bi_sender`` / ``bi_receiver`` (Algorithm 7).
* External task Cases E1-E4 (Algorithm 8): unambiguous because the aim
  stack, once drained, is itself always available as overflow capacity for
  the origin's blockers.

Documented simplification (internal task Cases I2/I3)
--------------------------------------------------------
For an *internal* task the origin and aim stack are the same stack, so there
is no second, already-cleared stack to dump overflow into while the target
container waits in its interim slot. The paper's literal pseudocode routes
this through an extra ``stmp1 -> stmp2 -> s+`` hand-off tied to an exact
slot-counting formula; reconstructed only from the printed algorithm boxes,
that hand-off has a retrieval ambiguity whenever the temp stack must also
receive part of the overflow (the target would be either strictly above or
buried under an unresolved order relative to that overflow). Rather than
risk silently mis-tracking which container is "the target" mid-procedure, we
use a construction that is provably retrieval-safe: the interim stack that
holds the target is *never* used as an overflow destination for the
remaining blockers (only the origin's other former neighbours are). This
still finds a plan on every instance we have stress-tested (see the
algorithm's test suite) and always terminates with the exact intended
final placement; in the rare knife-edge sub-case where the paper's own
7-step dance would have been strictly necessary to fit everything, this
construction reports the task as infeasible via ``InfeasibleInstance``
(caught gracefully by the outer heuristic) rather than mis-placing a
container.
"""

from __future__ import annotations

from typing import List, Tuple

from .feasibility import can_stabilize, compute_surplus, stable_height
from .state import (
    Move,
    State,
    apply_move,
    messiness,
    num_empty,
    stack_height,
    top_value,
    total_empty,
)


class InfeasibleInstance(Exception):
    pass


# ================================================================== #
# Algorithm 3: alpha                                                   #
# ================================================================== #

def alpha(state: State, s_plus: int, s_dst: int) -> int:
    others = [s for s in state.stacks if s != s_plus]
    positive = [num_empty(state, s) for s in others if num_empty(state, s) > 0]
    if not positive:
        raise InfeasibleInstance("alpha: no other stack has empty room")
    e_min = min(positive)
    e_plus = num_empty(state, s_plus)
    e_dst = num_empty(state, s_dst)
    E = total_empty(state)
    if e_dst > e_min:
        return E - e_plus - e_min - 1
    if e_min >= 2:
        return E - e_plus - e_min
    k_min = sum(1 for s in others if num_empty(state, s) == e_min)
    if k_min == 1:
        greater = [num_empty(state, s) for s in others if num_empty(state, s) > e_min]
        e_sec = min(greater) if greater else 0
        return E - e_plus - e_sec - 1
    return E - e_plus - 2


# ================================================================== #
# Algorithm 5: EvalMove                                                #
# ================================================================== #

def eval_move(state: State, s_src: int, s: int) -> Tuple[int, int]:
    c = state.stacks[s_src][-1]
    h_s = stack_height(state, s)
    sh_s = stable_height(state, s)
    if sh_s == h_s:
        if can_stabilize(state, s, c):
            cap_top = top_value(state, s)
            surplus = compute_surplus(state)
            affected = surplus.affected_demand(c, cap_top)
            return (1, affected)
        return (4, top_value(state, s))
    m_s = messiness(state, s, sh_s)
    if c >= m_s:
        return (2, c - m_s)
    return (3, m_s - c)


# ================================================================== #
# Algorithm 4: Relocate                                                #
# ================================================================== #

def relocate(state: State, s_src: int, k: int, R: List[int], moves: List[Move]) -> None:
    for _ in range(k):
        avail = [s for s in R if stack_height(state, s) < state.max_tiers]
        if not avail:
            raise InfeasibleInstance("relocate: no available destination")
        s_dst = min(avail, key=lambda s: eval_move(state, s_src, s))
        if not apply_move(state, s_src, s_dst):
            raise InfeasibleInstance(f"relocate: move ({s_src},{s_dst}) failed")
        moves.append((s_src, s_dst))


def raw_move(state: State, s_src: int, k: int, s_dst: int, moves: List[Move]) -> None:
    for _ in range(k):
        if not apply_move(state, s_src, s_dst):
            raise InfeasibleInstance(f"raw_move: ({s_src},{s_dst}) failed")
        moves.append((s_src, s_dst))


# ================================================================== #
# Algorithm 6: Interim / InterimFull                                   #
# ================================================================== #

def interim(state: State, I: List[int]) -> int:
    i1, i2 = [], []
    for s in I:
        h_s = stack_height(state, s)
        if h_s >= state.max_tiers:
            continue
        sh_s = stable_height(state, s)
        (i1 if sh_s < h_s else i2).append(s)
    if i1:
        return max(i1, key=lambda s: messiness(state, s, stable_height(state, s)))
    if i2:
        return min(i2, key=lambda s: top_value(state, s))
    raise InfeasibleInstance("interim: no candidate stack")


def interim_full(state: State, F: List[int]) -> int:
    f1, f2 = [], []
    for s in F:
        if stack_height(state, s) != state.max_tiers:
            continue
        fh = state.fixed_height[s]
        sh_s = stable_height(state, s)
        h_s = state.max_tiers
        if sh_s < h_s:
            f1.append(s)
        elif fh < sh_s == h_s:
            f2.append(s)
    if f1:
        return min(f1, key=lambda s: state.stacks[s][-1])
    if f2:
        return min(f2, key=lambda s: state.stacks[s][-1])
    raise InfeasibleInstance("interim_full: no candidate stack")


# ================================================================== #
# Algorithm 7: BiSender / BiReceiver                                   #
# ================================================================== #

def bi_sender(state: State, s_src1: int, k1: int, s_src2: int, k2: int, R: List[int], moves: List[Move]) -> None:
    while k1 + k2 > 0:
        if k1 == 0:
            relocate(state, s_src2, k2, R, moves)
            k2 = 0
        elif k2 == 0:
            relocate(state, s_src1, k1, R, moves)
            k1 = 0
        else:
            avail = [s for s in R if stack_height(state, s) < state.max_tiers]
            if not avail:
                raise InfeasibleInstance("bi_sender: no available destination")
            s_dst1 = min(avail, key=lambda s: eval_move(state, s_src1, s))
            v1 = eval_move(state, s_src1, s_dst1)
            s_dst2 = min(avail, key=lambda s: eval_move(state, s_src2, s))
            v2 = eval_move(state, s_src2, s_dst2)
            if v1 <= v2:
                if not apply_move(state, s_src1, s_dst1):
                    raise InfeasibleInstance("bi_sender: move 1 failed")
                moves.append((s_src1, s_dst1))
                k1 -= 1
            else:
                if not apply_move(state, s_src2, s_dst2):
                    raise InfeasibleInstance("bi_sender: move 2 failed")
                moves.append((s_src2, s_dst2))
                k2 -= 1


def bi_receiver(state: State, s_src: int, k1: int, R1: List[int], k2: int, R2: List[int], moves: List[Move]) -> None:
    while k1 + k2 > 0:
        if k1 == 0:
            relocate(state, s_src, k2, R2, moves)
            k2 = 0
        elif k2 == 0:
            relocate(state, s_src, k1, R1, moves)
            k1 = 0
        else:
            avail1 = [s for s in R1 if stack_height(state, s) < state.max_tiers]
            avail2 = [s for s in R2 if stack_height(state, s) < state.max_tiers]
            if not avail1 or not avail2:
                raise InfeasibleInstance("bi_receiver: no available destination")
            s_dst1 = min(avail1, key=lambda s: eval_move(state, s_src, s))
            v1 = eval_move(state, s_src, s_dst1)
            s_dst2 = min(avail2, key=lambda s: eval_move(state, s_src, s))
            v2 = eval_move(state, s_src, s_dst2)
            if v1 <= v2:
                if not apply_move(state, s_src, s_dst1):
                    raise InfeasibleInstance("bi_receiver: move 1 failed")
                moves.append((s_src, s_dst1))
                k1 -= 1
            else:
                if not apply_move(state, s_src, s_dst2):
                    raise InfeasibleInstance("bi_receiver: move 2 failed")
                moves.append((s_src, s_dst2))
                k2 -= 1


# ================================================================== #
# Task classification + move-count formulas (used by task_selection)  #
# ================================================================== #

def blocking_counts_internal(state: State, s_plus: int, t_plus: int) -> Tuple[int, int]:
    b1 = stack_height(state, s_plus) - 1 - t_plus
    b2 = t_plus - state.fixed_height[s_plus]
    return b1, b2


def blocking_counts_external(state: State, s_plus: int, t_plus: int, s_star: int) -> Tuple[int, int]:
    b1 = stack_height(state, s_plus) - 1 - t_plus
    b2 = stack_height(state, s_star) - state.fixed_height[s_star]
    return b1, b2


def internal_task_case(state: State, s_plus: int, b2: int) -> Tuple[str, int]:
    others = [s for s in state.stacks if s != s_plus]
    positive = [num_empty(state, s) for s in others if num_empty(state, s) > 0]
    if not positive:
        raise InfeasibleInstance("internal task: no other stack has room")
    e_min = min(positive)
    a = total_empty(state) - num_empty(state, s_plus) - e_min
    if a >= b2:
        return "I1", a
    n_others_with_room = sum(1 for s in others if stack_height(state, s) < state.max_tiers)
    return ("I2" if n_others_with_room > 1 else "I3"), a


def internal_task_move_count(state: State, s_plus: int, t_plus: int) -> int:
    b1, b2 = blocking_counts_internal(state, s_plus, t_plus)
    if b1 == 0 and b2 == 0:
        return 0
    case, _ = internal_task_case(state, s_plus, b2)
    total_unfixed = stack_height(state, s_plus) - state.fixed_height[s_plus]
    return total_unfixed + (1 if case == "I1" else 2)


def external_task_case(state: State, s_plus: int, s_star: int, b1: int, b2: int) -> Tuple[str, int]:
    a = total_empty(state) - num_empty(state, s_plus) - num_empty(state, s_star)
    if a >= b1 + b2:
        return "E1", a
    if b1 + 1 <= a < b1 + b2:
        return "E2", a
    if 1 <= a < b1 + min(1, b2):
        return "E3", a
    return "E4", a


def external_task_move_count(state: State, s_plus: int, t_plus: int, s_star: int) -> int:
    b1, b2 = blocking_counts_external(state, s_plus, t_plus, s_star)
    case, a = external_task_case(state, s_plus, s_star, b1, b2)
    if case == "E1":
        return b1 + b2 + 1
    if case == "E2":
        return b1 + b2 + 2
    if case == "E3":
        return 2 * b1 + b2 - a + 3
    return 2 * b1 + b2 + 4


# ================================================================== #
# Internal task accomplishment (Algorithm 2)                           #
# ================================================================== #

def accomplish_internal_task(state: State, s_plus: int, t_plus: int, moves: List[Move]) -> None:
    b1, b2 = blocking_counts_internal(state, s_plus, t_plus)
    if b1 == 0 and b2 == 0:
        return  # immediate task; nothing to do

    others = [s for s in state.stacks if s != s_plus]
    case, _ = internal_task_case(state, s_plus, b2)

    if case == "I1":
        for _ in range(b1):
            R = [s for s in others if stack_height(state, s) < state.max_tiers and alpha(state, s_plus, s) >= b2]
            if not R:
                raise InfeasibleInstance("I1: no valid destination under alpha filter")
            relocate(state, s_plus, 1, R, moves)
        I = [
            s for s in others
            if stack_height(state, s) < state.max_tiers
            and total_empty(state) - num_empty(state, s_plus) - num_empty(state, s) >= b2
        ]
        if not I:
            raise InfeasibleInstance("I1: no interim stack candidate")
        s_tmp = interim(state, I)
        raw_move(state, s_plus, 1, s_tmp, moves)
        relocate(state, s_plus, b2, [s for s in state.stacks if s not in (s_plus, s_tmp)], moves)
        raw_move(state, s_tmp, 1, s_plus, moves)
        return

    if case == "I2":
        relocate(state, s_plus, b1, others, moves)
        s_tmp1 = interim(state, others)
        raw_move(state, s_plus, 1, s_tmp1, moves)
        relocate(state, s_plus, b2, [s for s in state.stacks if s not in (s_plus, s_tmp1)], moves)
        raw_move(state, s_tmp1, 1, s_plus, moves)
        return

    # I3: exactly one other non-full stack.
    others_with_room = [s for s in others if stack_height(state, s) < state.max_tiers]
    s_prime = others_with_room[0]
    F = [s for s in state.stacks if s not in (s_plus, s_prime)]
    s_tmp = interim_full(state, F)
    bi_sender(state, s_tmp, 1, s_plus, b1, [s_prime], moves)
    raw_move(state, s_plus, 1, s_tmp, moves)
    relocate(state, s_plus, b2, [s for s in state.stacks if s not in (s_plus, s_tmp)], moves)
    raw_move(state, s_tmp, 1, s_plus, moves)


# ================================================================== #
# External task accomplishment (Algorithm 8)                           #
# ================================================================== #

def accomplish_external_task(state: State, s_plus: int, t_plus: int, s_star: int, moves: List[Move]) -> None:
    b1, b2 = blocking_counts_external(state, s_plus, t_plus, s_star)
    case, a = external_task_case(state, s_plus, s_star, b1, b2)
    others = [s for s in state.stacks if s not in (s_plus, s_star)]

    if case == "E1":
        bi_sender(state, s_plus, b1, s_star, b2, others, moves)
        raw_move(state, s_plus, 1, s_star, moves)
        return

    if case == "E2":
        k2 = a - 1 - b1
        bi_sender(state, s_plus, b1, s_star, k2, others, moves)
        s_tmp = interim(state, others)
        raw_move(state, s_plus, 1, s_tmp, moves)
        raw_move(state, s_star, b2 - k2, s_plus, moves)
        raw_move(state, s_tmp, 1, s_star, moves)
        return

    if case == "E3":
        k1 = a - 1
        bi_receiver(state, s_plus, k1, others, b1 - k1, [s_star], moves)
        s_tmp = interim(state, others)
        raw_move(state, s_plus, 1, s_tmp, moves)
        raw_move(state, s_star, (b1 - k1) + b2, s_plus, moves)
        raw_move(state, s_tmp, 1, s_star, moves)
        return

    # E4: a == 0.
    s_tmp = interim_full(state, others)
    bi_sender(state, s_tmp, 1, s_plus, b1, [s_star], moves)
    raw_move(state, s_plus, 1, s_tmp, moves)
    raw_move(state, s_star, b1 + b2 + 1, s_plus, moves)
    raw_move(state, s_tmp, 1, s_star, moves)


def accomplish_task(state: State, c_pos, s_star: int, moves: List[Move]) -> None:
    s_plus, t_plus = c_pos
    if s_plus == s_star:
        accomplish_internal_task(state, s_plus, t_plus, moves)
    else:
        accomplish_external_task(state, s_plus, t_plus, s_star, moves)
