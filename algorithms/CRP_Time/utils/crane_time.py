"""
Crane time of one move under f2 or f2vert of Voß & Schwarze (2019), the two
time models the CRP-Time algorithms minimise. The values come from the shared
evaluator in core/objectives.py, so a plan's time summed move by move equals
its evaluated time.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Sequence

from core.objectives import ObjectiveSpec, movement_f2, movement_f2_vertical
from core.plan import Movement


def check_objective(spec: ObjectiveSpec, algorithm: str, modes: Sequence[str] = ("crane_time",)) -> None:
    """Refuse every time model other than f2 and f2_vertical, and every objective mode not in modes.
    An algorithm passes "relocations" in modes only if it keeps its paper's relocation objective
    for the validation in the paper's own setting."""
    if spec.mode not in modes or spec.time_model not in ("f2", "f2_vertical"):
        raise ValueError(
            f"{algorithm} minimises {' or '.join(modes)} with the time model f2 or f2_vertical, "
            f"not objective {spec.mode!r} with time model {spec.time_model!r}"
        )


def move_time(move: Movement, spec: ObjectiveSpec) -> float:
    """Crane time of one move with its tiers set, under the selected time model.
    It does not depend on where the crane was before the move."""
    return movement_f2(move, spec) if spec.time_model == "f2" else movement_f2_vertical(move, spec)
