from pathlib import Path
import pytest
from crp_online.io.caserta_loader import load_all_caserta_instances
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


# Exact grand totals across all 840 Caserta benchmark instances matching C++ source code
EXPECTED_TOTALS = {
    "TLP": 36024,
    "RI": 29524,
    "MM": 25859,
    "PR4": 25787,
    "RDH": 26066,  # Verified against C++ oracle (Preprint Table 4 has erratum 26866)
    "ERI": 29524,
    "EM": 24612,
    "EGH": 30990,
    "PU1": 25049,
    "PU2": 24962,
}


@pytest.fixture(scope="module")
def all_instances():
    return load_all_caserta_instances(DATA_DIR)


def test_tlp_grand_total(all_instances):
    tot = sum(
        sum(1 for m in solve_restricted(Environment(inst), select_tlp_destination) if not m.is_retrieval)
        for inst in all_instances.values()
    )
    assert tot == EXPECTED_TOTALS["TLP"]


def test_ri_grand_total(all_instances):
    tot = sum(
        sum(1 for m in solve_restricted(Environment(inst), select_ri_destination) if not m.is_retrieval)
        for inst in all_instances.values()
    )
    assert tot == EXPECTED_TOTALS["RI"]


def test_minmax_grand_total(all_instances):
    tot = sum(
        sum(1 for m in solve_restricted(Environment(inst), select_minmax_destination) if not m.is_retrieval)
        for inst in all_instances.values()
    )
    assert tot == EXPECTED_TOTALS["MM"]


def test_pr4_grand_total(all_instances):
    tot = sum(
        sum(1 for m in solve_restricted(Environment(inst), select_pr4_destination) if not m.is_retrieval)
        for inst in all_instances.values()
    )
    assert tot == EXPECTED_TOTALS["PR4"]


def test_rdh_grand_total(all_instances):
    tot = sum(
        sum(1 for m in solve_restricted(Environment(inst), select_rdh_destination) if not m.is_retrieval)
        for inst in all_instances.values()
    )
    assert tot == EXPECTED_TOTALS["RDH"]


def test_eri_grand_total(all_instances):
    tot = sum(
        sum(1 for m in solve_restricted(Environment(inst), select_eri_destination) if not m.is_retrieval)
        for inst in all_instances.values()
    )
    assert tot == EXPECTED_TOTALS["ERI"]


def test_em_grand_total(all_instances):
    tot = sum(
        sum(1 for m in solve_restricted(Environment(inst), select_em_destination) if not m.is_retrieval)
        for inst in all_instances.values()
    )
    assert tot == EXPECTED_TOTALS["EM"]


def test_egh_grand_total(all_instances):
    tot = sum(
        sum(1 for m in solve_egh(Environment(inst)) if not m.is_retrieval)
        for inst in all_instances.values()
    )
    assert tot == EXPECTED_TOTALS["EGH"]


def test_pu1_grand_total(all_instances):
    tot = sum(
        sum(1 for m in solve_pu1(Environment(inst)) if not m.is_retrieval)
        for inst in all_instances.values()
    )
    assert tot == EXPECTED_TOTALS["PU1"]


def test_pu2_grand_total(all_instances):
    tot = sum(
        sum(1 for m in solve_pu2(Environment(inst)) if not m.is_retrieval)
        for inst in all_instances.values()
    )
    assert tot == EXPECTED_TOTALS["PU2"]
