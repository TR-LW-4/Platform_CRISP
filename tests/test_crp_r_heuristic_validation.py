"""Offline CRP-R heuristics retain solutions that the problem can validate."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from algorithms.CRP_R.heuristic.caserta_2009_lah.scoring import (
    greedy_trajectory,
    run_trajectory,
)
from algorithms.CRP_R.heuristic.caserta_2011_cm.algorithm import CasertaCM
from algorithms.CRP_R.heuristic.jovanovic_2019_aco.scoring import run_greedy_rbrp
from algorithms.CRP_R.heuristic.lan_2013.algorithm import LANHeuristic
from algorithms.CRP_R.heuristic.lan_2013.planner import build_lan_plan
from core.base_problem import ProblemConfig
from problems.CRP_R import CRP_R


class CRPRHeuristicValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.layout = Path(self.tmp.name) / "data2-2-1.dat"
        self.layout.write_text("2 2\n2 1 2\n0\n", encoding="utf-8")
        self.cfg = ProblemConfig(
            num_bays=2,
            num_rows=1,
            max_tiers=4,
            num_containers=2,
            num_groups=1,
            extra={"layout_file_path": str(self.layout)},
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def initial_state(self):
        env = CRP_R(self.cfg)
        env.reset(options={"skip_auto_retrieve": True})
        keys = [env._action_to_stack(i) for i in range(2)]
        stacks = {
            key: [int(c.priority) for c in env.yard.stacks[key].containers]
            for key in keys
        }
        return env, keys, stacks

    def test_lan_n1_plan_is_strictly_valid(self) -> None:
        self.assertEqual(LANHeuristic.config_schema()["N"]["max"], 1)
        env, _, _ = self.initial_state()
        plan = build_lan_plan(
            env.yard, list(env.containers), N=1, max_tiers=self.cfg.max_tiers
        )
        metrics = env.validate_plan(plan)
        self.assertEqual(metrics["feasible"], 1.0)
        self.assertEqual(metrics["relocations"], plan.num_relocations())

    def test_lah_greedy_and_restart_sequences_are_replayable(self) -> None:
        env, keys, stacks = self.initial_state()
        greedy_cost, greedy_dsts = greedy_trajectory(
            stacks, 2, self.cfg.max_tiers, keys
        )
        greedy_metrics = env.validate_actions(
            [keys.index(dst) for dst in greedy_dsts]
        )
        self.assertEqual(greedy_metrics["relocations"], greedy_cost)
        self.assertEqual(greedy_metrics["feasible"], 1.0)

        restart_cost, restart_dsts = run_trajectory(
            stacks,
            2,
            self.cfg.max_tiers,
            keys,
            np.random.RandomState(0),
            greedy_cost + 1,
        )
        restart_metrics = env.validate_actions(
            [keys.index(dst) for dst in restart_dsts]
        )
        self.assertEqual(restart_metrics["relocations"], restart_cost)
        self.assertEqual(restart_metrics["feasible"], 1.0)

    def test_aco_greedy_seed_keeps_physical_destinations(self) -> None:
        env, keys, stacks = self.initial_state()
        seed, cost = run_greedy_rbrp(
            stacks,
            keys,
            2,
            self.cfg.max_tiers,
            {key: i + 1 for i, key in enumerate(keys)},
        )
        actions = [keys.index(dst_key) for _, _, _, _, dst_key in seed]
        metrics = env.validate_actions(actions)
        self.assertEqual(metrics["relocations"], cost)
        self.assertEqual(metrics["feasible"], 1.0)

    def test_cm_machine_readable_path_parser(self) -> None:
        count, records = CasertaCM._parse_cm_output(
            "\n".join([
                "CM_MOVES_BEGIN",
                "CM_MOVE 2 0 1",
                "CM_MOVE 1 0 -1",
                "CM_MOVE 2 1 -1",
                "CM_MOVES_END",
                "CM : Solution found with 1 moves.",
            ])
        )
        self.assertEqual(count, 1)
        self.assertEqual(records, [(2, 0, 1), (1, 0, -1), (2, 1, -1)])


if __name__ == "__main__":
    unittest.main()
