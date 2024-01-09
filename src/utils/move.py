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
