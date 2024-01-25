import pytest

from utils.puzzle_pytorch import PuzzlePyTorch


@pytest.fixture
def puzzle():
    return PuzzlePyTorch(
        "cube_2/2/2",
        tuple("A;A;A;A;B;B;B;B;C;C;C;C;D;D;D;D;E;E;E;E;F;F;F;F".split(";")),
        0,
    )


def test_state(puzzle: PuzzlePyTorch):
    assert (
        puzzle.state_adapter.state_tensor_to_str(puzzle.solution_state_tensor.squeeze())
        == puzzle.solution_state
    )


def test_random_walk(puzzle: PuzzlePyTorch):
    BATCH = 4
    NUM_STEPS = 10
    for step, (states, moves) in enumerate(
        puzzle.random_walk_tensor(puzzle.solution_state_tensor.repeat(BATCH, 1), NUM_STEPS)
    ):
        assert states.shape == (BATCH, 24)
        assert len(list(moves)) == BATCH
    assert step == NUM_STEPS - 1 # type: ignore
