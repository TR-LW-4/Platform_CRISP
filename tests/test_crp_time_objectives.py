import unittest

from core.container import Container
from core.objectives import (
    KinematicsModel,
    ObjectiveSpec,
    annotate_plan_tiers,
    compute_f2,
    compute_f2_vertical,
    evaluate_plan_objectives,
)
from core.plan import Movement, RelocationPlan
from core.yard import Yard


class CRPTimeObjectiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.yard = Yard(num_bays=2, num_rows=1, max_tiers=4)
        self.yard.place(1, 1, Container(id=1, priority=1))
        self.yard.place(1, 1, Container(id=2, priority=2))
        self.plan = RelocationPlan([
            Movement(2, (1, 1), (2, 1)),
            Movement(1, (1, 1), None),
            Movement(2, (2, 1), None),
        ])
        self.spec = ObjectiveSpec(
            mode="crane_time",
            time_model="f2",
            stack_s_per_stack=1.2,
            pickup_place_s=30.0,
            empty_vertical_s_per_tier=2.59,
            loaded_vertical_s_per_tier=5.18,
            outside_height=1.5,
            max_tiers=4,
            num_rows=1,
        )

    def test_tier_replay(self) -> None:
        annotated = annotate_plan_tiers(self.yard, self.plan)
        tiers = [
            (move.from_tier, move.to_tier)
            for move in annotated.movements
        ]
        self.assertEqual(tiers, [(2, 1), (1, None), (1, None)])

    def test_paper_f2(self) -> None:
        annotated = annotate_plan_tiers(self.yard, self.plan)
        self.assertAlmostEqual(compute_f2(annotated, self.spec), 99.6)

    def test_paper_f2_vertical(self) -> None:
        annotated = annotate_plan_tiers(self.yard, self.plan)
        self.assertAlmostEqual(
            compute_f2_vertical(annotated, self.spec),
            180.54,
        )

    def test_weighted_objective_and_selected_alias(self) -> None:
        spec = ObjectiveSpec(
            **{
                **self.spec.__dict__,
                "mode": "weighted",
                "time_model": "f2_vertical",
                "relocation_weight": 10.0,
                "time_weight": 0.5,
            }
        )
        metrics = evaluate_plan_objectives(
            self.plan,
            spec,
            kinematics=KinematicsModel(),
            initial_yard=self.yard,
        )
        self.assertEqual(metrics["relocations"], 1.0)
        self.assertEqual(
            metrics["crane_time"],
            metrics["crane_time_vertical"],
        )
        self.assertAlmostEqual(
            metrics["objective_value"],
            10.0 + 0.5 * 180.54,
        )

    def test_zero_weight_pair_is_rejected(self) -> None:
        spec = ObjectiveSpec(
            mode="weighted",
            relocation_weight=0.0,
            time_weight=0.0,
        )
        with self.assertRaises(ValueError):
            spec.validate()


if __name__ == "__main__":
    unittest.main()
