"""
Crane time under f2 or f2vert of Voß & Schwarze (2019), the two time models
the CRP-Time algorithms minimise: the objective specification, the time of one
move, and the metrics of a plan. Copied from the shared evaluator in
core/objectives.py with the same order of float operations, so a plan's time
summed move by move equals its evaluated time and the platform's.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from core.plan import Movement, RelocationPlan


@dataclass(frozen=True)
class ObjectiveSpec:
    """User-selected CRP-Time objective and paper-model parameters."""

    mode: str = "crane_time"
    time_model: str = "f2"
    relocation_weight: float = 0.0
    time_weight: float = 1.0
    stack_s_per_stack: float = 1.2
    pickup_place_s: float = 30.0
    empty_vertical_s_per_tier: float = 2.59
    loaded_vertical_s_per_tier: float = 5.18
    outside_height: float = 1.5
    max_tiers: int = 3
    num_rows: int = 1

    @classmethod
    def from_config(cls, config) -> "ObjectiveSpec":
        extra = getattr(config, "extra", {}) or {}
        spec = cls(
            mode=str(extra.get("objective_mode", "crane_time")),
            time_model=str(extra.get("time_model", "f2")),
            relocation_weight=float(extra.get("relocation_weight", 1.0)),
            time_weight=float(extra.get("time_weight", 1.0)),
            stack_s_per_stack=float(
                extra.get(
                    "stack_s_per_stack",
                    extra.get("trolley_s_per_row", 1.2),
                )
            ),
            pickup_place_s=float(
                extra.get("pickup_place_s", extra.get("spreader_s", 30.0))
            ),
            empty_vertical_s_per_tier=float(
                extra.get("empty_vertical_s_per_tier", 2.59)
            ),
            loaded_vertical_s_per_tier=float(
                extra.get("loaded_vertical_s_per_tier", 5.18)
            ),
            outside_height=float(extra.get("outside_height", 1.5)),
            max_tiers=int(getattr(config, "max_tiers", 3)),
            num_rows=int(getattr(config, "num_rows", 1)),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.mode not in {"relocations", "crane_time", "weighted"}:
            raise ValueError(f"unknown objective_mode: {self.mode!r}")
        if self.time_model not in {"f2", "f2_vertical", "rmgc_current"}:
            raise ValueError(f"unknown time_model: {self.time_model!r}")
        if self.relocation_weight < 0 or self.time_weight < 0:
            raise ValueError("objective weights must be non-negative")
        if (
            self.mode == "weighted"
            and self.relocation_weight == 0
            and self.time_weight == 0
        ):
            raise ValueError("weighted objective requires at least one positive weight")

    @property
    def time_metric(self) -> str:
        return {
            "f2": "crane_time_f2",
            "f2_vertical": "crane_time_vertical",
            "rmgc_current": "crane_time_rmgc",
        }[self.time_model]


# Practice preset of the f2vert type: simulated ASC models with a row pitch of
# 2.54 m (trolley 1.0 and 1.2 m/s, times averaged), a tier height of 2.591 m and
# a spreader at 1.5 m/s empty and 0.75 m/s laden, rounded to 0.001 s. It has no
# value for tpp, so f2 is not reported for it.
PRACTICE_PARAMETERS: Dict[str, float] = {
    "stack_s_per_stack": 2.328,
    "empty_vertical_s_per_tier": 1.727,
    "loaded_vertical_s_per_tier": 3.455,
    "outside_height": 1.5,
}


def is_practice(values: Mapping[str, Any]) -> bool:
    """True for f2vert with the practice parameters (config values or an ObjectiveSpec's fields)."""
    if values.get("time_model") != "f2_vertical":
        return False
    try:
        return all(float(values[key]) == value for key, value in PRACTICE_PARAMETERS.items())
    except (KeyError, TypeError, ValueError):
        return False


def check_objective(spec: ObjectiveSpec, algorithm: str, modes: Sequence[str] = ("crane_time",)) -> None:
    """Refuse every time model other than f2 and f2_vertical, and every objective mode not in modes.
    An algorithm passes "relocations" in modes only if it keeps its paper's relocation objective
    for the validation in the paper's own setting."""
    if spec.mode not in modes or spec.time_model not in ("f2", "f2_vertical"):
        raise ValueError(
            f"{algorithm} minimises {' or '.join(modes)} with the time model f2 or f2_vertical, "
            f"not objective {spec.mode!r} with time model {spec.time_model!r}"
        )


def _stack_index(pos: Tuple[int, int], num_rows: int) -> int:
    """Flatten the platform coordinate to the paper's 1-based stack index."""
    bay, row = pos
    return (int(bay) - 1) * max(int(num_rows), 1) + int(row)


def movement_f2(move: Movement, spec: ObjectiveSpec) -> float:
    """Contribution of one movement to Voß--Schwarze f2."""
    src = _stack_index(move.from_pos, spec.num_rows)
    if move.is_retrieval:
        return 2.0 * spec.stack_s_per_stack * src + spec.pickup_place_s
    dst = _stack_index(move.to_pos, spec.num_rows)
    return (
        2.0 * spec.stack_s_per_stack * abs(src - dst)
        + spec.pickup_place_s
    )


def movement_f2_vertical(move: Movement, spec: ObjectiveSpec) -> float:
    """Contribution of one tier-annotated movement to f2vert."""
    h_max = float(spec.max_tiers + 1)
    tr = spec.empty_vertical_s_per_tier + spec.loaded_vertical_s_per_tier
    if move.from_tier is None:
        raise ValueError("f2_vertical requires source tiers")
    src = _stack_index(move.from_pos, spec.num_rows)
    if move.is_retrieval:
        vertical = 2.0 * h_max - move.from_tier - spec.outside_height
        return 2.0 * spec.stack_s_per_stack * src + vertical * tr
    if move.to_tier is None:
        raise ValueError("f2_vertical requires destination tiers")
    dst = _stack_index(move.to_pos, spec.num_rows)
    vertical = 2.0 * h_max - move.from_tier - move.to_tier
    return (
        2.0 * spec.stack_s_per_stack * abs(src - dst)
        + vertical * tr
    )


def move_time(move: Movement, spec: ObjectiveSpec) -> float:
    """Crane time of one move with its tiers set, under the selected time model.
    It does not depend on where the crane was before the move."""
    return movement_f2(move, spec) if spec.time_model == "f2" else movement_f2_vertical(move, spec)


def objective_value(metrics: Dict[str, float], spec: ObjectiveSpec) -> float:
    """Scalar objective selected by *spec* (lower is better)."""
    relocations = float(metrics["relocations"])
    selected_time = float(metrics[spec.time_metric])
    if spec.mode == "relocations":
        return relocations
    if spec.mode == "crane_time":
        return selected_time
    return (
        spec.relocation_weight * relocations
        + spec.time_weight * selected_time
    )


def evaluate(plan: RelocationPlan, spec: ObjectiveSpec) -> Dict[str, float]:
    """Relocations, f2, f2vert and the selected objective of a plan with its tiers set
    (utils.plan.replay sets them); a move without its tiers raises. The platform's
    evaluate_plan_objectives without the legacy RMGC time; f2 is left out for the
    practice preset.
    Example:
        metrics = evaluate(replay(stacks, positions, moves), spec)
    """
    for k, move in enumerate(plan.movements):
        if move.from_tier is None or (not move.is_retrieval and move.to_tier is None):
            raise ValueError(f"evaluate: move {k} (container {move.container_id}) has no tiers")
    metrics = {
        "relocations": float(plan.num_relocations()),
        "crane_time_f2": float(sum(movement_f2(move, spec) for move in plan.movements)),
        "crane_time_vertical": float(sum(movement_f2_vertical(move, spec) for move in plan.movements)),
        "total_moves": float(plan.num_moves()),
    }
    metrics["crane_time"] = metrics[spec.time_metric]
    metrics["time"] = metrics["crane_time"]
    metrics["objective_value"] = objective_value(metrics, spec)
    if is_practice({key: getattr(spec, key) for key in ("time_model", *PRACTICE_PARAMETERS)}):
        del metrics["crane_time_f2"]
    return metrics
