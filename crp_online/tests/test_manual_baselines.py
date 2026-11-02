from pathlib import Path
import pytest
from crp_online.io.caserta_loader import load_scenario_instances
from crp_online.core.environment import Environment
from crp_online.heuristics.restricted import solve_restricted
from crp_online.heuristics.manual import (
    select_tlp_destination,
    select_ri_destination,
    select_minmax_destination,
    select_pr4_destination,
    select_rdh_destination,
    select_eri_destination,
    select_em_destination,
    solve_egh,
    solve_pu1,
    solve_pu2,
)


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "caserta"


def _check_scenario(h: int, s: int, expected_em: float, expected_egh: float):
    instances = load_scenario_instances(DATA_DIR, height=h, stacks=s)
    assert len(instances) == 40

    em_moves_total = 0
    egh_moves_total = 0

    for inst in instances:
        env_em = Environment(inst)
        solve_restricted(env_em, select_em_destination)
        em_moves_total += env_em.total_relocations

        env_egh = Environment(inst)
        solve_egh(env_egh)
        egh_moves_total += env_egh.total_relocations

    em_mean = em_moves_total / 40.0
    egh_mean = egh_moves_total / 40.0

    print(f"\ndata{h}-{s} EM: {em_mean:.4f} (target: {expected_em:.4f}), EGH: {egh_mean:.4f} (target: {expected_egh:.4f})")
    assert abs(em_mean - expected_em) < 1e-4
    assert abs(egh_mean - expected_egh) < 1e-4


def test_scenario_data3_3():
    _check_scenario(3, 3, 5.0750, 5.6250)


def test_scenario_data5_4():
    _check_scenario(5, 4, 16.7500, 20.1750)


def test_scenario_data6_6():
    _check_scenario(6, 6, 35.9000, 43.6000)


def test_scenario_data10_6():
    _check_scenario(10, 6, 101.2500, 130.9750)


def test_scenario_data10_10():
    _check_scenario(10, 10, 139.2750, 193.8500)
