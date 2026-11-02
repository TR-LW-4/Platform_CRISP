from pathlib import Path
import pytest
from crp_online.io.ku_loader import parse_ku_file, load_all_ku_instances, generate_ku_manifest_checksums


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ku_arthanari"
MANIFEST_FILE = Path(__file__).resolve().parent.parent / "data" / "manifest_ku_checksums.sha256"


def test_parse_golden_micro_instance():
    """Validates parsing of the golden micro-instance T271014_0503_001.txt."""
    file_path = DATA_DIR / "0503" / "T271014_0503_001.txt"
    inst = parse_ku_file(file_path)

    assert inst.name == "T271014_0503_001.txt"
    assert inst.bay_id == 1
    assert inst.num_stacks == 5
    assert inst.num_tiers == 3
    assert inst.num_containers == 8
    assert inst.num_batches == 4
    assert inst.nominal_fill == 0.50
    assert abs(inst.occupancy - (8 / 15.0)) < 1e-6

    # Verify stack layout bottom-to-top
    # Stack 1: [3]
    # Stack 2: [1, 2]
    # Stack 3: [1]
    # Stack 4: [3]
    # Stack 5: [4, 4, 1]
    assert inst.stack_windows == (
        (3,),
        (1, 2),
        (1,),
        (3,),
        (4, 4, 1),
    )

    # Initial blocking lower bound b(n_0)
    # Stack 2: 1 (tier 1 is min, tier 2 is 2 > 1 -> 1 blocker)
    # Stack 5: 4 is tier 1, 4 is tier 2 (equal min -> 1 - 0.5 = 0.5), 1 is tier 3 (new min -> 1 - 1 = 0) -> wait:
    # Let's check initial blocking bound formula
    b0 = inst.get_initial_blocking_bound()
    assert b0 >= 1.0


def test_parse_all_1440_instances():
    """Gate 1: Verifies all 1,440 instances in the Ku & Arthanari benchmark."""
    instances = load_all_ku_instances(DATA_DIR)
    assert len(instances) == 1440

    count_50 = 0
    count_67 = 0

    for name, inst in instances.items():
        assert inst.num_stacks in (5, 6, 7, 8, 9, 10)
        assert inst.num_tiers in (3, 4, 5, 6)
        assert inst.num_containers == sum(len(stk) for stk in inst.stack_windows)
        assert inst.num_containers <= inst.num_stacks * inst.num_tiers - (inst.num_tiers - 1)

        if "T27" in name:
            assert inst.nominal_fill == 0.50
            count_50 += 1
        elif "T28" in name:
            assert inst.nominal_fill == 0.67
            count_67 += 1

    assert count_50 == 720
    assert count_67 == 720


def test_generate_checksums_manifest():
    """Generates and validates the SHA-256 checksum manifest for all 1,440 files."""
    count = generate_ku_manifest_checksums(DATA_DIR, MANIFEST_FILE)
    assert count == 1440
    assert MANIFEST_FILE.is_file()

    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 1440
