from pathlib import Path
import pytest
from crp_online.io.caserta_loader import load_scenario_instances
from crp_online.experiments.runner import BenchmarkRunner
from crp_online.experiments.statistics import compute_distribution_stats, is_equivalent


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "caserta"


def test_benchmark_runner_scenario():
    instances = load_scenario_instances(DATA_DIR, height=3, stacks=3)
    runner = BenchmarkRunner()

    result = runner.run_benchmark(
        condition_name="Test data3-3 LA=1.0",
        heuristics=["TLP", "MM", "EM"],
        instances=instances,
    )

    assert result.num_instances == 40
    assert "TLP" in result.totals
    assert "MM" in result.totals
    assert "EM" in result.totals

    # EM total on data3-3 is 203 (mean 5.0750)
    assert result.totals["EM"] == 203
    assert abs(result.means["EM"] - 5.0750) < 1e-4


def test_statistics_and_equivalence():
    vals = [24000.0, 24050.0, 23980.0, 24020.0, 24010.0]
    stats = compute_distribution_stats(vals)

    assert abs(stats["median"] - 24010.0) < 1e-5
    assert stats["iqr"] >= 0.0

    # Equivalence within 2% band
    assert is_equivalent(stats["median"], 24004.0, rel_tol=0.02)
    assert not is_equivalent(30000.0, 24004.0, rel_tol=0.02)
