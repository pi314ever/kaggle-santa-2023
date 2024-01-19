import random
import torch
import torch.nn as nn
from torch import Tensor

from utils.typing import STATE_TYPE

from .puzzle import Puzzle
from .move import apply_move_tensor
from .state import StateAdapter


class PuzzlePyTorch(Puzzle):
    """Puzzle Class for PyTorch that implements operations with native PyTorch tensors as state instead of tuple[str, ...]

    State tensors are of shape (b, n) or (n, ) where n is length of puzzle and b is batch size.
    """

    def __init__(
        self,
        puzzle_type: str,
        solution_state: STATE_TYPE,
        wildcards: int = 0,
        random=random.Random(),
    ):
        super().__init__(puzzle_type, solution_state, wildcards, random)

        self.state_adapter = StateAdapter(tuple(sorted(set(solution_state))))
        self._solution_state_tensor = self.state_adapter.state_str_to_tensor(
            self.solution_state
        )

    @property
    def solution_state_tensor(self):
        return self._solution_state_tensor

    def is_solved(self, state: Tensor):
        return torch.sum(state == self.solution_state_tensor, dim=1) <= self.wildcards

    def move_state(self, state: Tensor, move_str):
        return apply_move_tensor(state, self.moves.move_from_str(move_str))

    def random_walk(self, states: Tensor, num_steps: int) -> Tensor:
        """Tensors of shape (n, b) where n is length of puzzle and b is batch size."""
        raise NotImplementedError
