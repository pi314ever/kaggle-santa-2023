import random

from .data import PUZZLE_INFO_DF
from .move import apply_move
from .typing import STATE_TYPE


class PuzzleMoves:
    def __init__(self, puzzle_type: str):
        self.puzzle_type = puzzle_type
        allowed_moves_dict = PUZZLE_INFO_DF[
            PUZZLE_INFO_DF["puzzle_type"] == puzzle_type
        ]["allowed_moves"].iloc[0]
        self.allowed_moves: list[str] = list(allowed_moves_dict.keys())
        self.allowed_moves.sort()
        self.moves: list[STATE_TYPE] = [
            allowed_moves_dict[move] for move in self.allowed_moves
        ]
        self._move_to_index = {move: i for i, move in enumerate(self.allowed_moves)}

    @property
    def num_allowed_moves(self):
        return len(self.allowed_moves)

    def opposite_move_str(self, move_str: str):
        """Get the opposite move."""
        if move_str not in self._move_to_index:
            raise ValueError(
                f"{move_str} is not a valid move. Valid moves: {self.allowed_moves}"
            )
        if move_str.startswith("-"):
            return move_str[1:]
        return "-" + move_str

    def move_from_str(self, move_str):
        """Return the move from a string."""
        if move_str not in self._move_to_index:
            raise ValueError(
                f"{move_str} is not a valid move. Valid moves: {self.allowed_moves}"
            )
        return self.moves[self._move_to_index[move_str]]

    def move_from_idx(self, move_idx: int):
        """Return the move from a string."""
        return self.moves[move_idx]


class Puzzle:
    def __init__(
        self,
        puzzle_type: str,
        solution_state: STATE_TYPE,
        wildcards: int = 0,
        random=random.Random(),
    ):
        self.puzzle_type = puzzle_type
        self.moves = PuzzleMoves(puzzle_type)
        self.solution_state = solution_state
        self.wildcards = wildcards
        self.random = random

    @property
    def allowed_moves(self):
        return self.moves.allowed_moves

    @property
    def num_allowed_moves(self):
        return self.moves.num_allowed_moves

    def is_solved(self, state):
        """Check if the puzzle is approximately solved."""
        assert len(state) == len(
            self.solution_state
        ), f"State given is length {len(state)}, but solution is length {len(self.solution_state)}"

        dist = sum(int(s != ss) for s, ss in zip(state, self.solution_state))

        return dist <= self.wildcards

    def move_state(self, state, move_str):
        """Apply a move to the puzzle state."""
        return apply_move(state, self.moves.move_from_str(move_str))

    def random_walk(self, states: list[STATE_TYPE], num_steps: int) -> list[STATE_TYPE]:
        """Randomly walk the puzzle states."""
        raise NotImplementedError

    def is_valid_solution(self, state: STATE_TYPE, moves: list[str]):
        """Check if a solution is valid."""
        for move in moves:
            state = self.move_state(state, move)
        return self.is_solved(state)

    def heuristic(self, state: STATE_TYPE, moves: list[str]):
        """Heuristic for the puzzle."""
        return len(moves) + sum(
            int(s != ss) for s, ss in zip(state, self.solution_state)
        )
