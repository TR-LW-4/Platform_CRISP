"""
Crane kinematics model and objective functions for BRP-Fixed /
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

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .plan import Movement, RelocationPlan


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
