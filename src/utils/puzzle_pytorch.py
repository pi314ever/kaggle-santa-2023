import random

import torch
from torch import Tensor

from utils.typing import STATE_TYPE

from .move import apply_move_tensor
from .puzzle import Puzzle
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

    def set_device(self, device: torch.device):
        self._solution_state_tensor = self._solution_state_tensor.to(device)

    def is_solved_tensor(self, state: Tensor):
        return torch.sum(state == self.solution_state_tensor, dim=1) <= self.wildcards

    def move_state_tensor(self, state: Tensor, move_str):
        return apply_move_tensor(state, self.moves.move_from_str(move_str))

    def random_walk_single(self, state: Tensor, num_steps: int):
        """Tensors of shape (n,) where n is length of puzzle."""
        assert state.ndim == 1, f"States must be 1-dimensional. Got {state.ndim}"
        step = 0
        state = state.unsqueeze(0)
        while step < num_steps:
            move = self.random.sample(self.allowed_moves, 1)[0]
            state = self.move_state_tensor(state, move)
            step += 1

            yield state.squeeze(), move

    def random_walk_tensor(self, states: Tensor, num_steps: int):
        """Tensors of shape (n, b) where n is length of puzzle and b is batch size."""
        assert states.ndim == 2, f"States must be 2-dimensional. Got {states.ndim}"
        step = 0
        while step < num_steps:
            shuffled_moves = self.random.sample(
                self.allowed_moves, len(self.allowed_moves)
            )
            shuffled_groups = self.random.sample(
                range(states.shape[0]), states.shape[0]
            )
            for i, move in enumerate(shuffled_moves):
                subset = shuffled_groups[
                    i
                    * states.shape[0]
                    // len(shuffled_moves) : (i + 1)
                    * states.shape[0]
                    // len(shuffled_moves)
                ]
                states[subset, :] = self.move_state_tensor(states[subset, :], move)
            step += 1

            yield states, zip(shuffled_moves, shuffled_groups)
