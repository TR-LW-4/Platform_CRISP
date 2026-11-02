from pathlib import Path
import pytest
from crp_online.io.caserta_loader import load_all_caserta_instances
from crp_online.experiments.runner import BenchmarkRunner


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "caserta"


# Table 8 (Restricted, Group Size GS = 2) Ground Truth from Đurasević et al. (2026)
TABLE_8_TARGETS = {
    "unknown": {
        "ERI": 29739,
        "EM": 28143,
        "RDH": 27999,
        "EGH": 30375,
    },
    "known": {
        "ERI": 29557,
        "EM": 27133,
        "RDH": 27682,
        "EGH": 30369,
    },
}


@pytest.fixture(scope="module")
def all_instances():
    instances_map = load_all_caserta_instances(DATA_DIR)
    return list(instances_map.values())


def test_table8_gs2_unknown_parity(all_instances):
    """Gate 0: Verifies deterministic manual heuristics on Caserta under GS=2 Unknown within group."""
    runner = BenchmarkRunner()
    heuristics = ["ERI", "EM", "RDH", "EGH"]

    res = runner.run_benchmark(
        condition_name="Table 8: GS=2 Unknown",
        heuristics=heuristics,
        instances=all_instances,
        use_container_groups=True,
        group_size=2,
        known_within_group=False,
    )

    print("\n--- Gate 0: Table 8 GS=2 Unknown Results ---")
    for h in heuristics:
        tot = res.totals[h]
        expected = TABLE_8_TARGETS["unknown"][h]
        print(f"{h}: got {tot}, expected {expected} (diff: {tot - expected:+d})")

    for h in heuristics:
        tot = res.totals[h]
        expected = TABLE_8_TARGETS["unknown"][h]
        assert tot == expected, f"{h} under GS=2 Unknown expected {expected}, got {tot}"


def test_table8_gs2_known_parity(all_instances):
    """Gate 0: Verifies deterministic manual heuristics on Caserta under GS=2 Known within group."""
    runner = BenchmarkRunner()
    heuristics = ["ERI", "EM", "RDH", "EGH"]

    res = runner.run_benchmark(
        condition_name="Table 8: GS=2 Known",
        heuristics=heuristics,
        instances=all_instances,
        use_container_groups=True,
        group_size=2,
        known_within_group=True,
    )

    print("\n--- Gate 0: Table 8 GS=2 Known Results ---")
    for h in heuristics:
        tot = res.totals[h]
        expected = TABLE_8_TARGETS["known"][h]
        print(f"{h}: got {tot}, expected {expected} (diff: {tot - expected:+d})")

    for h in heuristics:
        tot = res.totals[h]
        expected = TABLE_8_TARGETS["known"][h]
        assert tot == expected, f"{h} under GS=2 Known expected {expected}, got {tot}"
