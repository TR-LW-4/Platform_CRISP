"""Strict solution validation for the restricted block relocation problem."""

from __future__ import annotations

import tempfile
import unittest
import math
from pathlib import Path

from core.base_problem import ProblemConfig
from core.plan import Movement, RelocationPlan
from problems.CRP_R import CRP_R


class CRPRValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.layout = Path(self.tmp.name) / "data2-2-1.dat"
        # Bottom -> top: stack 1 is [1, 2], stack 2 is empty.
        self.layout.write_text("2 2\n2 1 2\n0\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def make_env(self) -> CRP_R:
        return CRP_R(ProblemConfig(
            num_bays=2,
            num_rows=1,
            max_tiers=4,
            num_containers=2,
            num_groups=1,
            extra={"layout_file_path": str(self.layout)},
        ))

    def test_validate_actions_accepts_complete_restricted_solution(self) -> None:
        env = self.make_env()
        metrics = env.validate_actions([1])

        self.assertEqual(metrics["feasible"], 1.0)
        self.assertEqual(metrics["completed"], 1.0)
        self.assertEqual(metrics["validated"], 1.0)
        self.assertEqual(metrics["relocations"], 1.0)
        self.assertEqual(metrics["validation_conflicts"], 0.0)
        self.assertEqual(env.get_last_validation_errors(), [])

    def test_validate_actions_rejects_invalid_and_incomplete_solutions(self) -> None:
        env = self.make_env()
        invalid = env.validate_actions([0])
        self.assertEqual(invalid["feasible"], 0.0)
        self.assertTrue(math.isinf(invalid["relocations"]))
        self.assertTrue(env.get_last_validation_errors())

        incomplete = env.validate_actions([])
        self.assertEqual(incomplete["feasible"], 0.0)
        self.assertIn("ended before", env.get_last_validation_errors()[0])

        out_of_range = env.validate_actions([2])
        self.assertEqual(out_of_range["feasible"], 0.0)
        self.assertIn("outside", env.get_last_validation_errors()[0])

    def test_validate_plan_accepts_complete_restricted_plan(self) -> None:
        env = self.make_env()
        plan = RelocationPlan([
            Movement(2, (1, 1), (2, 1)),
            Movement(1, (1, 1), None),
            Movement(2, (2, 1), None),
        ])
        metrics = env.validate_plan(plan)

        self.assertEqual(metrics["feasible"], 1.0)
        self.assertEqual(metrics["relocations"], 1.0)
        self.assertEqual(metrics["retrievals"], 2.0)
        self.assertEqual(metrics["validation_conflicts"], 0.0)

    def test_validate_plan_rejects_cleaning_move_for_crp_r(self) -> None:
        env = self.make_env()
        # C2 is the current target's blocker; first move it legally, then try
        # moving it again from an unrelated stack instead of retrieving C1.
        plan = RelocationPlan([
            Movement(2, (1, 1), (2, 1)),
            Movement(2, (2, 1), (1, 1)),
        ])
        metrics = env.validate_plan(plan)

        self.assertEqual(metrics["feasible"], 0.0)
        self.assertTrue(math.isinf(metrics["relocations"]))
        self.assertIn("not from current target stack", env.get_last_validation_errors()[0])

    def test_validate_plan_requires_priority_order_and_completion(self) -> None:
        env = self.make_env()
        wrong_retrieval = RelocationPlan([
            Movement(2, (1, 1), (2, 1)),
            Movement(2, (2, 1), None),
        ])
        metrics = env.validate_plan(wrong_retrieval)

        self.assertEqual(metrics["feasible"], 0.0)
        self.assertIn("expected priority 1", env.get_last_validation_errors()[0])


if __name__ == "__main__":
    unittest.main()
