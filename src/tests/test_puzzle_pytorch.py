import pytest

from utils.puzzle_pytorch import PuzzlePyTorch


@pytest.fixture
def puzzle():
    return PuzzlePyTorch("test", ("a", "b", "c"), 0)
