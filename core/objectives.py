"""
Crane kinematics model and objective functions for CRP-R /
Lee & Lee (2010) style crane time estimation.

The KinematicsModel computes w(crane_pos, move): the time an RMGC needs
to complete `move` given that the crane's spreader last stopped at `crane_pos`.

Default parameters match Lee & Lee (2010), Section 4
-----------------------------------------------------
  gantry speed : 3.5  s / bay
  trolley speed: 1.2  s / row (one container width)
  gantry accel : 40   s  combined acceleration + deceleration overhead
  spreader     : 30   s  combined pickup + place-down

The gantry (along-track) and trolley (cross-track) axes of an RMGC move
simultaneously but independently, so repositioning time is the maximum of
the two travel times.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .plan import Movement, RelocationPlan
    from .yard import Yard


# Position of the truck that picks up retrieved containers.
# Modelled at one end of the crane track: bay 0, row 0.
_TRUCK_POS: Tuple[int, int] = (0, 0)


# ================================================================ #
#  Kinematics model                                                  #
# ================================================================ #

@dataclass
class KinematicsModel:
    """
    RMGC time model.  All parameters and outputs in seconds.

    Attributes
    ----------
    gantry_s_per_bay  : travel time per bay in the gantry (along-track) axis
    trolley_s_per_row : travel time per row in the trolley (cross-track) axis
    gantry_accel_s    : combined acceleration + deceleration overhead,
                        added once when the gantry actually moves
    spreader_s        : combined pickup + place-down time for the spreader
    """
    gantry_s_per_bay:  float = 3.5
    trolley_s_per_row: float = 1.2
    gantry_accel_s:    float = 40.0
    spreader_s:        float = 30.0

    # ---------------------------------------------------------------- #
    # Factory                                                            #
    # ---------------------------------------------------------------- #

    @classmethod
    def from_config_extra(cls, extra: Dict) -> "KinematicsModel":
        """
        Build from ProblemConfig.extra dict.
        The GUI stores kinematics keys directly in extra (not nested), so we
        look for them at the top level, then fall back to default values.
        """
        return cls(
            gantry_s_per_bay  = float(extra.get("gantry_s_per_bay",  3.5)),
            trolley_s_per_row = float(extra.get("trolley_s_per_row", 1.2)),
            gantry_accel_s    = float(extra.get("gantry_accel_s",   40.0)),
            spreader_s        = float(extra.get("spreader_s",        30.0)),
        )

    # ---------------------------------------------------------------- #
    # Core calculation                                                   #
    # ---------------------------------------------------------------- #

    def travel_time(
        self,
        pos_a: Tuple[int, int],
        pos_b: Tuple[int, int],
    ) -> float:
        """
        Time for the crane to reposition from pos_a to pos_b.

        Gantry and trolley move simultaneously, so total time =
        max(gantry_time, trolley_time).
        Gantry acceleration overhead is added only when the gantry moves.
        """
        bay_a, row_a = pos_a
        bay_b, row_b = pos_b
        bay_diff = abs(bay_b - bay_a)
        row_diff = abs(row_b - row_a)

        gantry_t = bay_diff * self.gantry_s_per_bay
        if bay_diff > 0:
            gantry_t += self.gantry_accel_s

        trolley_t = row_diff * self.trolley_s_per_row

        return max(gantry_t, trolley_t)

    def move_time(
        self,
        crane_pos: Tuple[int, int],
        move:      "Movement",
    ) -> float:
        """
        w(crane_pos, move): time to complete *move* given crane is at crane_pos.

        Sequence
        --------
        1. Reposition crane from crane_pos → move.from_pos
        2. Pickup (spreader_s / 2)
        3. Carry crane from move.from_pos → move.to_pos (or truck)
        4. Place-down (spreader_s / 2)
        """
        to_pos = move.to_pos if move.to_pos is not None else _TRUCK_POS

        t_reposition = self.travel_time(crane_pos, move.from_pos)
        t_carry      = self.travel_time(move.from_pos, to_pos)
        t_spreader   = self.spreader_s   # pickup + place-down combined

        return t_reposition + t_carry + t_spreader

    def end_pos(self, move: "Movement") -> Tuple[int, int]:
        """Position of the crane spreader after completing *move*."""
        return move.to_pos if move.to_pos is not None else _TRUCK_POS


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


def _stack_index(pos: Tuple[int, int], num_rows: int) -> int:
    """Flatten the platform coordinate to the paper's 1-based stack index."""
    bay, row = pos
    return (int(bay) - 1) * max(int(num_rows), 1) + int(row)


def annotate_plan_tiers(
    initial_yard: "Yard",
    plan: "RelocationPlan",
) -> "RelocationPlan":
    """Replay *plan* and attach 1-based source/destination tiers to every move."""
    from .plan import Movement, RelocationPlan

    yard = copy.deepcopy(initial_yard)
    annotated = RelocationPlan()
    for move in plan.movements:
        src = yard.stacks.get(move.from_pos)
        if (
            src is None
            or src.is_empty
            or (
                src.top.id != move.container_id
                and src.top.priority != move.container_id
            )
        ):
            raise ValueError(
                f"cannot recover tier for container {move.container_id} "
                f"at source {move.from_pos}"
            )
        from_tier = src.height
        container = src.pop()
        if move.is_retrieval:
            annotated.add(
                Movement(
                    container.id,
                    move.from_pos,
                    None,
                    from_tier=from_tier,
                    to_tier=None,
                )
            )
            continue

        dst = yard.stacks.get(move.to_pos)
        if dst is None or dst.is_full:
            raise ValueError(f"cannot recover tier at destination {move.to_pos}")
        dst.push(container)
        annotated.add(
            Movement(
                container.id,
                move.from_pos,
                move.to_pos,
                from_tier=from_tier,
                to_tier=dst.height,
            )
        )
    return annotated


def movement_f2(move: "Movement", spec: ObjectiveSpec) -> float:
    """Contribution of one movement to Voß--Schwarze f2."""
    src = _stack_index(move.from_pos, spec.num_rows)
    if move.is_retrieval:
        return 2.0 * spec.stack_s_per_stack * src + spec.pickup_place_s
    dst = _stack_index(move.to_pos, spec.num_rows)
    return (
        2.0 * spec.stack_s_per_stack * abs(src - dst)
        + spec.pickup_place_s
    )


def compute_f2(plan: "RelocationPlan", spec: ObjectiveSpec) -> float:
    """Voß--Schwarze f2: horizontal travel plus fixed handling time."""
    return sum(movement_f2(move, spec) for move in plan.movements)


def movement_f2_vertical(move: "Movement", spec: ObjectiveSpec) -> float:
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


def compute_f2_vertical(plan: "RelocationPlan", spec: ObjectiveSpec) -> float:
    """Voß--Schwarze f2vert with tier-dependent pickup/place-down effort."""
    return sum(movement_f2_vertical(move, spec) for move in plan.movements)


def movement_selected_time(
    move: "Movement",
    spec: ObjectiveSpec,
    kinematics: KinematicsModel,
    crane_pos: Tuple[int, int] = (1, 1),
) -> Tuple[float, Tuple[int, int]]:
    """Selected time-model contribution and resulting legacy crane position."""
    if spec.time_model == "f2":
        value = movement_f2(move, spec)
    elif spec.time_model == "f2_vertical":
        value = movement_f2_vertical(move, spec)
    else:
        value = kinematics.move_time(crane_pos, move)
    return float(value), kinematics.end_pos(move)


def movement_objective_cost(
    move: "Movement",
    spec: ObjectiveSpec,
    kinematics: KinematicsModel,
    crane_pos: Tuple[int, int] = (1, 1),
) -> Tuple[float, float, Tuple[int, int]]:
    """Additive scalar objective, selected time, and resulting crane position."""
    time_cost, next_pos = movement_selected_time(
        move, spec, kinematics, crane_pos
    )
    relocation_cost = 0.0 if move.is_retrieval else 1.0
    if spec.mode == "relocations":
        scalar = relocation_cost
    elif spec.mode == "crane_time":
        scalar = time_cost
    else:
        scalar = (
            spec.relocation_weight * relocation_cost
            + spec.time_weight * time_cost
        )
    return float(scalar), float(time_cost), next_pos


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


def evaluate_plan_objectives(
    plan: "RelocationPlan",
    spec: ObjectiveSpec,
    *,
    kinematics: Optional[KinematicsModel] = None,
    initial_yard: Optional["Yard"] = None,
) -> Dict[str, float]:
    """Evaluate f1, f2, f2vert, legacy RMGC time, and the selected objective."""
    tier_plan = plan
    if any(move.from_tier is None for move in plan.movements):
        if initial_yard is None:
            raise ValueError("initial_yard is required to recover movement tiers")
        tier_plan = annotate_plan_tiers(initial_yard, plan)

    kin = kinematics or KinematicsModel()
    metrics = {
        "relocations": float(tier_plan.num_relocations()),
        "crane_time_f2": float(compute_f2(tier_plan, spec)),
        "crane_time_vertical": float(compute_f2_vertical(tier_plan, spec)),
        "crane_time_rmgc": float(compute_crane_time(tier_plan, kin)),
        "total_moves": float(tier_plan.num_moves()),
    }
    metrics["crane_time"] = metrics[spec.time_metric]
    metrics["time"] = metrics["crane_time"]
    metrics["objective_value"] = objective_value(metrics, spec)
    return metrics


# ================================================================ #
#  Plan-level objective                                              #
# ================================================================ #

def compute_crane_time(
    plan:        "RelocationPlan",
    kinematics:  KinematicsModel,
    initial_pos: Tuple[int, int] = (1, 1),
) -> float:
    """
    Total crane working time (seconds) for an entire RelocationPlan.

    Parameters
    ----------
    plan        : ordered movement sequence
    kinematics  : RMGC time model
    initial_pos : crane starting position (default: stack (bay=1, row=1))
    """
    total     = 0.0
    crane_pos = initial_pos

    for move in plan.movements:
        total    += kinematics.move_time(crane_pos, move)
        crane_pos = kinematics.end_pos(move)

    return total


# ================================================================ #
#  Lower bound helper                                                #
# ================================================================ #

def lower_bound_relocations(yard: "Yard") -> int:  # noqa: F821
    """
    Standard BRP lower bound: total number of bad overlaps in the yard.

    A bad overlap is a (lower, upper) pair within one stack where
    lower.priority < upper.priority (a later-retrieved container sits
    on top of an earlier-retrieved one, forcing at least one relocation).

    This is a per-stack-additive lower bound used in Lee & Lee (2010).
    """
    return yard.total_bad_overlaps()
