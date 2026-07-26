"""
Stage 4 (Sec. 3.4): stack filling, and the four competing filling policies
(the Hf set): None, Standard, Safe, Stop.

After a container is well-located onto s*, it sits at the top of s*. Filling
exploits any remaining empty tiers of s* by greedily adding accessible
not-well-located containers (highest due-date value first, since it must
stay <= the current top to remain well located) from elsewhere in the bay.

* None: no filling.
* Standard: fill greedily until s* is full or no candidate remains
  (Exposito-Izquierdo et al. 2012's original policy).
* Safe: only commit the (simulated) standard fill if it leaves at most ``a``
  empty tiers in s* afterwards -- i.e. only do it if it would make s* "close
  to full" -- else skip filling entirely for this stack.
* Stop: a look-ahead variant of Standard that halts *before* placing a
  candidate ca (currently sitting directly above some cb in its own stack,
  with p(cb) > p(ca)) whenever doing so would foreclose the possibility of
  later well-locating cb onto the same destination (i.e. the destination's
  current top cs' already satisfies p(cs') > p(cb), so cb could still fit
  there if ca were not placed first).
"""

from __future__ import annotations

from typing import List, Tuple

from .core import Stacks, clone_stacks, is_well_located

Move = Tuple[int, int]

FILLING_HEURISTICS = ("None", "Standard", "Safe", "Stop")


def _move_top(stacks: Stacks, src: int, dst: int, max_tiers: int) -> bool:
    if src == dst or not stacks[src] or len(stacks[dst]) >= max_tiers:
        return False
    stacks[dst].append(stacks[src].pop())
    return True


def _fill_candidates(stacks: Stacks, s: int) -> List[Tuple[int, int]]:
    """Returns (priority, source_stack) pairs eligible to be filled into s."""
    if not stacks[s]:
        return []
    top = stacks[s][-1]
    out = []
    for k, arr in stacks.items():
        if k == s or not arr:
            continue
        v = arr[-1]
        if v > top:
            continue
        if is_well_located(stacks, (k, len(arr) - 1)):
            continue  # already fine where it is; don't disturb it
        out.append((v, k))
    return out


def standard_fill(stacks: Stacks, s: int, max_tiers: int, moves: List[Move]) -> None:
    while len(stacks[s]) < max_tiers:
        cands = _fill_candidates(stacks, s)
        if not cands:
            break
        cands.sort(key=lambda x: (-x[0], x[1]))
        src = cands[0][1]
        if not _move_top(stacks, src, s, max_tiers):
            break
        moves.append((src, s))


def safe_fill(stacks: Stacks, s: int, max_tiers: int, moves: List[Move], a: int) -> None:
    trial_stacks = clone_stacks(stacks)
    trial_moves: List[Move] = []
    standard_fill(trial_stacks, s, max_tiers, trial_moves)
    remaining_empty = max_tiers - len(trial_stacks[s])
    if remaining_empty > a:
        return  # not "safe" enough: skip filling this stack entirely
    for src, dst in trial_moves:
        stacks[dst].append(stacks[src].pop())
    moves.extend(trial_moves)


def stop_fill(stacks: Stacks, s: int, max_tiers: int, moves: List[Move]) -> None:
    while len(stacks[s]) < max_tiers:
        cands = _fill_candidates(stacks, s)
        if not cands:
            break
        cands.sort(key=lambda x: (-x[0], x[1]))
        src = cands[0][1]
        arr_src = stacks[src]
        ca = arr_src[-1]
        cb = arr_src[-2] if len(arr_src) >= 2 else None
        if cb is not None and cb > ca:
            cs_prime = stacks[s][-1] if stacks[s] else None
            if cs_prime is not None and cs_prime > cb:
                break
        if not _move_top(stacks, src, s, max_tiers):
            break
        moves.append((src, s))


def apply_filling(
    heuristic: str,
    stacks: Stacks,
    touched: List[int],
    max_tiers: int,
    moves: List[Move],
    safe_slack: int = 1,
) -> None:
    if heuristic == "None":
        return
    for s in touched:
        if s not in stacks or not stacks[s] or len(stacks[s]) >= max_tiers:
            continue
        if heuristic == "Standard":
            standard_fill(stacks, s, max_tiers, moves)
        elif heuristic == "Safe":
            safe_fill(stacks, s, max_tiers, moves, a=safe_slack)
        elif heuristic == "Stop":
            stop_fill(stacks, s, max_tiers, moves)
        else:
            raise ValueError(f"unknown filling heuristic: {heuristic}")
