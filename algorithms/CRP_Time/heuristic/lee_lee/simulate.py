"""
Replay a RelocationPlan and report stacking / over-height conflicts.

Used only by Lee & Lee (2010) phase 2 and phase 3, which rewrite an
existing plan and must check that the new sequence is still executable.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

from core.plan import RelocationPlan
from core.yard import Yard


class ConflictType(Enum):
    STACKING = "stacking"   # target container not on top of source stack
    OVERHIGH = "overhigh"   # destination stack would exceed max_tiers


@dataclass
class SimResult:
    feasible:        bool
    num_moves:       int
    num_relocations: int
    num_retrievals:  int
    conflicts: List[Tuple[int, ConflictType, str]]  # (step_idx, type, message)


def simulate_plan(
    yard:      Yard,
    plan:      RelocationPlan,
    max_tiers: int,
) -> SimResult:
    """
    Execute *plan* on a deep copy of *yard* and return SimResult.

    Simulation continues past conflicts (to collect all errors), treating
    conflicting moves as no-ops so subsequent moves can still be checked.
    The original yard is not mutated.
    """
    sim: Yard = copy.deepcopy(yard)
    conflicts: List[Tuple[int, ConflictType, str]] = []
    n_reloc    = 0
    n_retrieve = 0

    for step, move in enumerate(plan.movements):
        src_stack = sim.stacks.get(move.from_pos)

        stacking_ok = True
        if src_stack is None or src_stack.is_empty:
            conflicts.append((
                step, ConflictType.STACKING,
                f"Source {move.from_pos} is empty",
            ))
            stacking_ok = False
        elif src_stack.top.id != move.container_id:
            conflicts.append((
                step, ConflictType.STACKING,
                f"C{move.container_id} not on top of {move.from_pos} "
                f"(top is C{src_stack.top.id})",
            ))
            stacking_ok = False

        if not stacking_ok:
            continue

        container = src_stack.pop()

        if move.to_pos is None:
            n_retrieve += 1
        else:
            dst_stack = sim.stacks.get(move.to_pos)
            if dst_stack is None:
                conflicts.append((
                    step, ConflictType.OVERHIGH,
                    f"Destination {move.to_pos} not found",
                ))
                src_stack.push(container)
            elif dst_stack.height >= max_tiers:
                conflicts.append((
                    step, ConflictType.OVERHIGH,
                    f"Destination {move.to_pos} full ({dst_stack.height}/{max_tiers})",
                ))
                src_stack.push(container)
            else:
                dst_stack.push(container)
                n_reloc += 1

    return SimResult(
        feasible        = (len(conflicts) == 0),
        num_moves       = len(plan.movements),
        num_relocations = n_reloc,
        num_retrievals  = n_retrieve,
        conflicts       = conflicts,
    )
