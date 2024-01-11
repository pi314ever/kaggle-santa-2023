import numpy as np

from .data import PUZZLE_DF, PUZZLE_INFO_DF
from .move import apply_move
from .typing import STATE_TYPE


class PuzzleInstance:
    def __init__(
        self, puzzle_type: str, solution_state: STATE_TYPE, wildcards: int = 0
    ):
        self.puzzle_type = puzzle_type
        allowed_moves_dict = PUZZLE_INFO_DF[
            PUZZLE_INFO_DF["puzzle_type"] == puzzle_type
        ]["allowed_moves"].iloc[0]
        self.allowed_moves: list[str] = list(allowed_moves_dict.keys())
        self.allowed_moves.sort()  # Sort the moves to ensure consistency
        self.moves: list[STATE_TYPE] = [
            allowed_moves_dict[move] for move in self.allowed_moves
        ]
        self._index_to_move = {i: move for i, move in enumerate(self.allowed_moves)}
        self._move_to_index = {move: i for i, move in enumerate(self.allowed_moves)}
        self.solution_state = solution_state
        self.wildcards = wildcards

    def is_solved(self, state):
        """Check if the puzzle is approximately solved."""
        if self.solution_state is None:
            raise ValueError("No solution state is not set.")
        dist = sum(int(s != ss) for s, ss in zip(state, self.solution_state))

        return dist <= self.wildcards

    def move_from_str(self, move_str):
        """Return the move from a string."""
        return self.moves[self._move_to_index[move_str]]

    def move(self, state, move_str):
        """Apply a move to the puzzle state."""
        if move_str not in self._move_to_index:
            raise ValueError(
                f"{move_str} is not a valid move. Valid moves: {self.allowed_moves}"
            )
        self.state = apply_move(state, self.move_from_str(move_str))
