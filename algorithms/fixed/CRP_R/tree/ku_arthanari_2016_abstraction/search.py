"""Run Ku and Arthanari's abstraction search on one bay."""

from __future__ import annotations

import time
from typing import Callable, List, Optional, Sequence, Tuple

from .abstraction_core import build_pdb, solve

Move = Tuple[int, int, Optional[int]]


def search(
    stacks_init: Sequence[Sequence[int]],
    n: int,
    max_tiers: int,
    time_limit: float,
    pdb_depth: int = 5,
    cache_depth: int = 20,
    max_cache_size: int = 1_000_000,
    pdb_build_time_limit_s: float = 60.0,
    stop: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[float, int], None]] = None,
) -> Tuple[List[Move], Optional[int]]:
    """Return the relocation plan and its relocation count."""
    started = time.perf_counter()
    stopper = _Stop(stop)
    build_limit = pdb_build_time_limit_s
    if time_limit is not None:
        build_limit = min(build_limit, time_limit)
    levels = build_pdb(
        depth=max(0, int(pdb_depth)),
        n_cols=len(stacks_init),
        m_tiers=max_tiers,
        time_limit_s=build_limit,
        stop_event=stopper,
    )
    if stopper.is_set() or time.perf_counter() - started >= time_limit:
        return [], None
    remaining = max(0.0, time_limit - (time.perf_counter() - started))
    result = solve(
        [list(stack) for stack in stacks_init],
        n,
        max_tiers,
        pdb_levels=levels,
        pdb_depth=max(levels),
        cache_depth=max(0, int(cache_depth)),
        max_cache_size=max(1, int(max_cache_size)),
        time_limit_s=remaining,
        stop_event=stopper,
    )
    if result["best"] < 0:
        return [], None
    moves = _replay(stacks_init, result["path"])
    if on_progress is not None:
        on_progress(time.perf_counter() - started, int(result["best"]))
    return moves, int(result["best"])


def _replay(stacks_init, path) -> List[Move]:
    state = [list(stack) for stack in stacks_init]
    moves = _retrieve(state)
    for source, destination in path:
        container = state[source][-1]
        state[source].pop()
        state[destination].append(container)
        moves.append((container, source, destination))
        moves.extend(_retrieve(state))
    return moves


def _retrieve(stacks) -> List[Move]:
    moves: List[Move] = []
    while True:
        present = [item for stack in stacks for item in stack]
        if not present:
            return moves
        target = min(present)
        source = next(index for index, stack in enumerate(stacks) if target in stack)
        if stacks[source][-1] != target:
            return moves
        stacks[source].pop()
        moves.append((target, source, None))


class _Stop:
    def __init__(self, stop: Optional[Callable[[], bool]]) -> None:
        self.stop = stop

    def is_set(self) -> bool:
        return self.stop is not None and self.stop()
