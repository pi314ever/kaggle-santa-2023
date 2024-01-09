# Source: https://www.kaggle.com/code/whats2000/a-star-algorithm-polytope-permutation
def apply_move(state, move, inverse=False):
    """Apply a move or its inverse to the puzzle state."""
    if inverse:
        return tuple(state[i] for i in inverse_move(move))
    else:
        return tuple(state[i] for i in move)


def inverse_move(move):
    """Return the inverse of a move."""
    tmp = {v: k for k, v in enumerate(move)}
    return [tmp[i] for i in range(len(move))]


def apply_move_tensor(state, move, inverse=False):
    """Apply a move or its inverse to the puzzle state."""
    if inverse:
        return state[inverse_move(move), :]
    else:
        return state[move, :]


def is_valid_solution(id: int, moves: list[str]):
    from .data import PUZZLE_DF, PUZZLE_INFO_DF

    """Check if a solution is valid."""
    puzzle_type = PUZZLE_DF.loc[id]["puzzle_type"]
    solution_state = PUZZLE_DF.loc[id]["solution_state"]
    state = PUZZLE_DF.loc[id]["initial_state"]
    wildcards = PUZZLE_DF.loc[id]["num_wildcards"]
    allowed_moves = PUZZLE_INFO_DF[PUZZLE_INFO_DF["puzzle_type"] == puzzle_type].iloc[
        0
    ]["allowed_moves"]

    for move in moves:
        state = apply_move(state, allowed_moves[move])
    return (
        sum(int(state[i] != solution_state[i]) for i in range(len(state))) <= wildcards
    )
