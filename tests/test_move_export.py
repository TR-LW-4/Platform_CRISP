"""Unit tests for move_history → plan / JSON / crane_time helpers."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Optional, Tuple

from core.move_export import (
    attach_crane_time,
    crane_time_for_yard,
    export_yard_moves,
    moves_from_final_record,
    plan_from_json,
    plan_from_moves,
)
from core.objectives import KinematicsModel, compute_crane_time
from core.plan import Movement


@dataclass
class _FakeMove:
    kind: str
    container_id: int
    src: Tuple[int, int]
    dst: Optional[Tuple[int, int]] = None
    from_tier: Optional[int] = None
    to_tier: Optional[int] = None


class _FakeYard:
    def __init__(self, history):
        self.move_history = history


class MoveExportTests(unittest.TestCase):
    def test_plan_and_json_roundtrip(self) -> None:
        moves = [
            _FakeMove("relocate", 3, (1, 1), (1, 2)),
            _FakeMove("retrieve", 1, (1, 1), None),
        ]
        plan = plan_from_moves(moves)
        self.assertEqual(plan.num_relocations(), 1)
        self.assertEqual(plan.num_retrievals(), 1)
        self.assertEqual(plan.movements[0], Movement(3, (1, 1), (1, 2)))
        self.assertEqual(plan.movements[1], Movement(1, (1, 1), None))

        payload = export_yard_moves(_FakeYard(moves))
        self.assertEqual(
            payload,
            [
                {"kind": "relocate", "container_id": 3, "from": [1, 1], "to": [1, 2]},
                {"kind": "retrieve", "container_id": 1, "from": [1, 1], "to": None},
            ],
        )
        again = plan_from_json(payload)
        self.assertEqual(again.movements, plan.movements)

    def test_crane_time_uses_kinematics(self) -> None:
        moves = [
            _FakeMove("relocate", 2, (1, 1), (1, 2)),
            _FakeMove("retrieve", 1, (1, 1), None),
        ]
        yard = _FakeYard(moves)
        extra = {
            "gantry_s_per_bay": 3.5,
            "trolley_s_per_row": 1.2,
            "gantry_accel_s": 40.0,
            "spreader_s": 30.0,
        }
        ct = crane_time_for_yard(yard, extra)
        expected = compute_crane_time(
            plan_from_moves(moves),
            KinematicsModel.from_config_extra(extra),
        )
        self.assertAlmostEqual(ct, expected)
        self.assertGreater(ct, 0.0)

        metrics = attach_crane_time({"relocations": 1.0}, yard, extra)
        self.assertEqual(metrics["relocations"], 1.0)
        self.assertAlmostEqual(metrics["crane_time"], ct)

    def test_optional_tiers_roundtrip(self) -> None:
        moves = [
            _FakeMove("relocate", 3, (1, 1), (1, 2), 3, 2),
            _FakeMove("retrieve", 1, (1, 1), None, 2, None),
        ]
        payload = export_yard_moves(_FakeYard(moves))
        self.assertEqual(payload[0]["from_tier"], 3)
        self.assertEqual(payload[0]["to_tier"], 2)
        self.assertEqual(payload[1]["from_tier"], 2)
        plan = plan_from_json(payload)
        self.assertEqual(plan.movements[0].from_tier, 3)
        self.assertEqual(plan.movements[0].to_tier, 2)
        self.assertEqual(plan.movements[1].from_tier, 2)

    def test_moves_from_final_record(self) -> None:
        payload = [{"kind": "retrieve", "container_id": 1, "from": [1, 1], "to": None}]
        self.assertEqual(
            moves_from_final_record({"extra": {"moves": payload}}),
            payload,
        )
        self.assertIsNone(moves_from_final_record({"extra": {}}))
        self.assertIsNone(moves_from_final_record(None))


if __name__ == "__main__":
    unittest.main()
