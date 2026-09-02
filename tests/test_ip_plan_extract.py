"""Extract IP incumbents into RelocationPlan and match validate_plan counts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from algorithms.CRP_R.solver_ip.common import (
    occupancy_stages_to_plan,
    plan_from_brp_ii_vars,
    sequential_retrieval_plan,
)
from algorithms.CRP_R.solver_ip.tanaka_voss_2022_ip.algorithm import (
    TanakaVossIP2022,
    _plan_from_tanaka_relocations,
)
from core.base_problem import ProblemConfig
from problems.CRP_R import CRP_R


def _tiny_cfg(layout: Path) -> ProblemConfig:
    return ProblemConfig(
        num_bays=2,
        num_rows=1,
        max_tiers=4,
        num_containers=2,
        num_groups=1,
        extra={"layout_file_path": str(layout)},
    )


class IPPlanExtractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.layout = Path(self.tmp.name) / "data2-2-1.dat"
        self.layout.write_text("2 2\n2 1 2\n0\n", encoding="utf-8")
        self.sorted_layout = Path(self.tmp.name) / "data2-2-sorted.dat"
        self.sorted_layout.write_text("2 2\n1 1\n1 2\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _env(self, layout: Path) -> CRP_R:
        env = CRP_R(_tiny_cfg(layout))
        env.reset(options={"skip_auto_retrieve": True})
        return env

    def test_brp_ii_x_y_plan_matches_validate_plan(self) -> None:
        env = self._env(self.layout)
        # Relocate 2 from stack 1 tier 2 to stack 2; retrieve 1 then 2.
        plan = plan_from_brp_ii_vars(
            env.yard,
            x_vals=((1, 2, 2, 1, 2, 1),),
            y_vals=((1, 1, 1),),
            n=2,
        )
        metrics = env.validate_plan(plan)
        self.assertEqual(metrics["feasible"], 1.0)
        self.assertEqual(metrics["relocations"], 1.0)
        self.assertGreater(metrics["crane_time"], 0.0)
        self.assertEqual(metrics["time"], metrics["crane_time"])

    def test_occupancy_diff_plan_matches_validate_plan(self) -> None:
        env = self._env(self.layout)
        occupancy = {
            1: {1: (1, 1), 2: (1, 2)},
            2: {2: (2, 1)},
        }
        keys = [(1, 1), (2, 1)]
        pri_to_id = {1: 1, 2: 2}
        plan = occupancy_stages_to_plan(
            occupancy, {(1, 2): 1}, 2, keys, pri_to_id,
        )
        metrics = env.validate_plan(plan)
        self.assertEqual(metrics["feasible"], 1.0)
        self.assertEqual(metrics["relocations"], 1.0)

    def test_sequential_retrieval_plan_for_sorted_yard(self) -> None:
        env = self._env(self.sorted_layout)
        plan = sequential_retrieval_plan(env.yard)
        metrics = env.validate_plan(plan)
        self.assertEqual(metrics["feasible"], 1.0)
        self.assertEqual(metrics["relocations"], 0.0)

    def test_tanaka_stdout_relocation_line_and_plan(self) -> None:
        line = "Relocation 1: [  2] 1->2"
        recs = [
            (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            for m in TanakaVossIP2022._RELOC_RE.finditer(line)
        ]
        self.assertEqual(recs, [(2, 1, 2)])
        env = self._env(self.layout)
        plan = _plan_from_tanaka_relocations(env.yard, recs)
        metrics = env.validate_plan(plan)
        self.assertEqual(metrics["feasible"], 1.0)
        self.assertEqual(metrics["relocations"], 1.0)

    def test_caserta_gurobi_obj_matches_reconstructed_plan(self) -> None:
        try:
            import gurobipy  # noqa: F401
            from gurobipy import GurobiError
        except ImportError:
            self.skipTest("gurobipy not installed")
        from algorithms.CRP_R.solver_ip.caserta_2012_brp2.algorithm import (
            _initial_layout,
            _solve_brp_ii,
        )

        env = self._env(self.layout)
        try:
            result = _solve_brp_ii(
                _initial_layout(env.yard, 2), 2, 2, 4, 30.0, 0,
            )
        except GurobiError as exc:
            self.skipTest(f"Gurobi unavailable: {exc}")
        self.assertIsNotNone(result["obj"])
        plan = plan_from_brp_ii_vars(
            env.yard, result["x_vals"], result["y_vals"], 2,
        )
        metrics = env.validate_plan(plan)
        self.assertEqual(metrics["feasible"], 1.0)
        self.assertEqual(metrics["relocations"], float(result["obj"]))

    def test_wan_gurobi_obj_matches_reconstructed_plan(self) -> None:
        try:
            import gurobipy  # noqa: F401
            from gurobipy import GurobiError
        except ImportError:
            self.skipTest("gurobipy not installed")
        from algorithms.CRP_R.solver_ip.wan_2009.algorithm import _run_full_mrip

        env = self._env(self.layout)
        try:
            result = _run_full_mrip(env.yard, 2, 2, 4, 30.0, 0)
        except GurobiError as exc:
            self.skipTest(f"Gurobi unavailable: {exc}")
        self.assertIsNotNone(result["obj"])
        self.assertIsNotNone(result.get("plan"))
        metrics = env.validate_plan(result["plan"])
        self.assertEqual(metrics["feasible"], 1.0)
        self.assertEqual(metrics["relocations"], float(result["obj"]))

    def test_tang_gurobi_obj_matches_reconstructed_plan(self) -> None:
        try:
            import gurobipy  # noqa: F401
            from gurobipy import GurobiError
        except ImportError:
            self.skipTest("gurobipy not installed")
        from algorithms.CRP_R.solver_ip.tang_2015_ilp.algorithm import _run_full_ilp

        env = self._env(self.layout)
        try:
            result = _run_full_ilp(env.yard, 2, 2, 4, 30.0, 0)
        except GurobiError as exc:
            self.skipTest(f"Gurobi unavailable: {exc}")
        self.assertIsNotNone(result["obj"])
        self.assertIsNotNone(result.get("plan"))
        metrics = env.validate_plan(result["plan"])
        self.assertEqual(metrics["feasible"], 1.0)
        self.assertEqual(metrics["relocations"], float(result["obj"]))

    def test_galle_cs_equal_uses_geometry_fallback(self) -> None:
        env = self._env(self.layout)
        from algorithms.CRP_R.solver_ip.common import (
            occupancy_from_yard,
            occupancy_stages_to_plan,
            priority_to_container_id,
            stack_keys,
        )
        plan = occupancy_stages_to_plan(
            {1: occupancy_from_yard(env.yard)},
            {},
            2,
            stack_keys(env.yard),
            priority_to_container_id(env.yard),
            max_tiers=4,
        )
        metrics = env.validate_plan(plan)
        self.assertEqual(metrics["feasible"], 1.0)
        self.assertEqual(metrics["relocations"], 1.0)

    def test_galle_gurobi_obj_matches_reconstructed_plan(self) -> None:
        try:
            import gurobipy  # noqa: F401
            from gurobipy import GurobiError
        except ImportError:
            self.skipTest("gurobipy not installed")
        from algorithms.CRP_R.solver_ip.galle_2018.algorithm import _solve_ip
        from algorithms.CRP_R.solver_ip.common import (
            occupancy_from_yard,
            occupancy_stages_to_plan,
            priority_to_container_id,
            stack_keys,
        )

        env = self._env(self.layout)
        try:
            result = _solve_ip(
                env.yard, 2, 2, 4, 30.0, True, True, True, 8e10, 0,
            )
        except GurobiError as exc:
            self.skipTest(f"Gurobi unavailable: {exc}")
        self.assertIsNotNone(result["obj"])
        occupancy = result.get("occupancy") or {1: occupancy_from_yard(env.yard)}
        plan = occupancy_stages_to_plan(
            occupancy,
            result.get("y_vals") or {},
            2,
            stack_keys(env.yard),
            priority_to_container_id(env.yard),
            max_tiers=4,
        )
        metrics = env.validate_plan(plan)
        self.assertEqual(metrics["feasible"], 1.0)
        self.assertEqual(metrics["relocations"], float(result["obj"]))

    def test_demelo_flow_plan_from_y_z_k(self) -> None:
        from algorithms.CRP_R.solver_ip.de_melo_silva_2018.algorithm import (
            _plan_from_demelo_flow,
        )

        class _Bit:
            def __init__(self, on: bool) -> None:
                self.X = 1.0 if on else 0.0

        env = self._env(self.layout)
        y, z, k = {}, {}, {}
        T, G, S, H = 3, 2, 2, 4
        for t in range(1, T + 1):
            for g in range(1, G + 1):
                for s in range(1, S + 1):
                    for h in range(1, H + 1):
                        y[t, g, s, h] = _Bit(False)
                        z[t, g, s, h] = _Bit(False)
                for n in range(1, G + 1):
                    k[t, g, n] = _Bit(False)
        # t=1 relocate 2 from stack 1 to stack 2; t=2 retrieve 1; t=3 retrieve 2
        y[1, 2, 2, 1] = _Bit(True)
        z[1, 2, 1, 2] = _Bit(True)
        z[2, 1, 1, 1] = _Bit(True)
        k[2, 1, 1] = _Bit(True)
        z[3, 2, 2, 1] = _Bit(True)
        k[3, 2, 2] = _Bit(True)
        plan = _plan_from_demelo_flow(env.yard, y, z, k, T, G, S, H, "m1")
        metrics = env.validate_plan(plan)
        self.assertEqual(metrics["feasible"], 1.0)
        self.assertEqual(metrics["relocations"], 1.0)

    def test_bacci_period_layout_parser(self) -> None:
        from algorithms.CRP_R.solver_ip.bacci_2020.algorithm import (
            parse_bacci_period_layouts,
        )
        from algorithms.CRP_R.solver_ip.common import (
            occupancy_stages_to_plan,
            priority_to_container_id,
            stack_keys,
        )

        stdout = (
            "**** Time period = 1\n\n"
            "[2][ ]\n"
            "[1][ ]\n\n"
            "Reshuffles = 1\n"
            "Reshuffled blocks at time period 1 = 2\n\n"
            "**** Time period = 2\n\n"
            "[ ][2]\n"
            "[ ][ ]\n\n"
            "Reshuffles = 1\n"
        )
        layouts = parse_bacci_period_layouts(stdout, 2, 4)
        self.assertEqual(layouts[1][1], (1, 1))
        self.assertEqual(layouts[1][2], (1, 2))
        self.assertEqual(layouts[2][2], (2, 2))
        env = self._env(self.layout)
        plan = occupancy_stages_to_plan(
            layouts, {(1, 2): 1}, 2,
            stack_keys(env.yard),
            priority_to_container_id(env.yard),
            max_tiers=4,
        )
        metrics = env.validate_plan(plan)
        self.assertEqual(metrics["feasible"], 1.0)
        self.assertEqual(metrics["relocations"], 1.0)


if __name__ == "__main__":
    unittest.main()
