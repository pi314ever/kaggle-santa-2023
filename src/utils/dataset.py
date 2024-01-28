import torch
from torch.utils.data import IterableDataset

from .puzzle_pytorch import PuzzlePyTorch


class PuzzleWalkDataset(IterableDataset):
    def __init__(
        self,
        puzzle: PuzzlePyTorch,
        walk_length: int,
        num_walks: int,
    ):
        self.puzzle = puzzle
        self.walk_length = walk_length
        self.num_walks = num_walks

    def __iter__(self):
        # Random walk from a goal puzzle state
        for _ in range(self.num_walks):
            states = self.puzzle.solution_state_tensor.repeat(self.walk_length + 1, 1)

            i = 1
            for s, _ in self.puzzle.random_walk_single(
                states[i, :],
                self.walk_length,
            ):
                states[i] = s
                i += 1

            yield states, torch.arange(self.walk_length + 1)
