from pathlib import Path
import pytest
from crp_online.io.caserta_loader import parse_caserta_file, load_all_caserta_instances


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "caserta"


def test_load_single_instance():
    test_file = DATA_DIR / "data3-3-1.dat"
    inst = parse_caserta_file(test_file)
    assert inst.name == "data3-3-1"
    assert inst.num_stacks == 3
    assert inst.num_containers == 9
    assert inst.initial_height == 3
    assert inst.max_stack_height == 5
    assert len(inst.initial_layout) == 3
    # Stack 0: 3, 7, 1
    assert inst.initial_layout[0] == (3, 7, 1)
    # Stack 1: 2, 6, 5
    assert inst.initial_layout[1] == (2, 6, 5)
    # Stack 2: 8, 9, 4
    assert inst.initial_layout[2] == (8, 9, 4)


def test_all_840_caserta_instances_conservation():
    instances = load_all_caserta_instances(DATA_DIR)
    assert len(instances) == 840, f"Expected 840 Caserta benchmark instances, found {len(instances)}"

    for name, inst in instances.items():
        assert inst.max_stack_height == inst.initial_height + 2
        assert inst.num_containers == inst.num_stacks * inst.initial_height

        # Flatten container IDs
        all_ids = []
        for stack in inst.initial_layout:
            assert len(stack) == inst.initial_height
            all_ids.extend(stack)

        assert len(all_ids) == inst.num_containers
        assert sorted(all_ids) == list(range(1, inst.num_containers + 1)), (
            f"Instance {name} container IDs do not form permutation of 1..{inst.num_containers}"
        )
