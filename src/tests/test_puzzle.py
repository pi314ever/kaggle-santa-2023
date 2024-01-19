import pytest

from utils.puzzle import Puzzle


@pytest.fixture
def puzzle():
    return Puzzle(
        "cube_2/2/2",
        tuple("A;A;A;A;B;B;B;B;C;C;C;C;D;D;D;D;E;E;E;E;F;F;F;F".split(";")),
        0,
    )


def test_is_solved(puzzle: Puzzle):
    assert puzzle.is_solved(
        tuple("A;A;A;A;B;B;B;B;C;C;C;C;D;D;D;D;E;E;E;E;F;F;F;F".split(";"))
    )


def test_is_solved_with_wildcards(puzzle: Puzzle):
    puzzle.wildcards = 2
    assert puzzle.is_solved(
        tuple("A;A;A;A;B;B;B;B;C;C;C;C;D;D;D;D;E;E;E;E;F;F;F;F".split(";"))
    )
    assert puzzle.is_solved(
        tuple("A;A;A;B;A;B;B;B;C;C;C;C;D;D;D;D;E;E;E;E;F;F;F;F".split(";"))
    )
