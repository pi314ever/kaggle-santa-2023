from .data import PUZZLE_INFO_DF
from .move import apply_move


class Puzzle:
    def __init__(self, puzzle_type: str, initial_state: tuple, solution_state: tuple):
        self.puzzle_type = puzzle_type
        allowed_moves_dict = PUZZLE_INFO_DF[
            PUZZLE_INFO_DF["puzzle_type"] == puzzle_type
        ]["allowed_moves"].iloc[0]
        self.allowed_moves = list(allowed_moves_dict.keys())
        self.allowed_moves.sort()  # Sort the moves to ensure consistency
        self.moves = [allowed_moves_dict[move] for move in self.allowed_moves]
        self._index_to_move = {i: move for i, move in enumerate(self.allowed_moves)}
        self._move_to_index = {move: i for i, move in enumerate(self.allowed_moves)}
        self.initial_state = tuple(initial_state)  # Make a copy of the initial state
        self.solution_state = tuple(solution_state)

    def is_solved(self, state):
        """Check if the puzzle is solved."""
        return state == self.solution_state

    def is_approximately_solved(self, state, wildcards: int = 0):
        """Check if the puzzle is approximately solved."""
        return (
            sum(
                [
                    1 if state[i] != self.solution_state[i] else 0
                    for i in range(len(state))
                ]
            )
            <= wildcards
        )

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
