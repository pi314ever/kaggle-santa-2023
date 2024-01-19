import torch
from utils.move import apply_move_tensor, apply_move, inverse_move


def test_apply_move_tensor():
    state = torch.tensor([0, 1, 2], dtype=torch.uint8).unsqueeze(0)
    move = [2, 0, 1]
    expected_state = torch.tensor([2, 0, 1], dtype=torch.uint8).unsqueeze(0)
    moved_state = apply_move_tensor(state, move)
    assert torch.equal(moved_state, expected_state)
    assert moved_state.shape == (1, 3)
