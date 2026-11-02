from pathlib import Path
import pytest
from crp_online.io.caserta_loader import parse_caserta_file
from crp_online.core.environment import Environment
from crp_online.core.location import Location


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "caserta"


def test_environment_initialization():
    inst = parse_caserta_file(DATA_DIR / "data3-3-1.dat")
    env = Environment(inst)

    assert env.num_stacks == 3
    assert env.max_stack_height == 5
    assert env.num_containers == 9
    assert env.current_target == 1
    assert env.crane_pos == Location(bay=0, stack=-1)
    assert env.get_stack_of_container(1) == 0
    assert env.get_top_container(0) == 1
    assert env.get_stack_height(0) == 3
    assert not env.is_empty()


def test_environment_retrieve_and_relocate():
    inst = parse_caserta_file(DATA_DIR / "data3-3-1.dat")
    env = Environment(inst)

    # In data3-3-1: Stack 0 is (3, 7, 1). Container 1 is already at top!
    move = env.retrieve(0)
    assert move.is_retrieval
    assert move.container_id == 1
    assert move.source.stack == 0
    assert move.destination.stack == -1
    assert env.current_target == 2
    assert env.get_stack_height(0) == 2
    assert env.get_top_container(0) == 7

    # Current target is 2. In Stack 1: (2, 6, 5). Top is 5.
    assert env.get_stack_of_container(2) == 1
    assert env.get_top_container(1) == 5

    # Relocate top of Stack 1 (5) to Stack 2 (height 3)
    move2 = env.relocate(source_stack=1, dest_stack=2)
    assert not move2.is_retrieval
    assert move2.container_id == 5
    assert env.get_stack_height(1) == 2
    assert env.get_stack_height(2) == 4
    assert env.get_top_container(1) == 6
    assert env.get_top_container(2) == 5


def test_environment_illegal_moves():
    inst = parse_caserta_file(DATA_DIR / "data3-3-1.dat")
    env = Environment(inst)

    # Cannot relocate to same stack
    with pytest.raises(ValueError, match="same"):
        env.relocate(0, 0)

    # Cannot retrieve non-target container
    # Stack 1 has top 5, but target is 1
    with pytest.raises(ValueError, match="current target"):
        env.retrieve(1)
