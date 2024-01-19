import pytest
import torch

from utils.state import StateAdapter


@pytest.fixture
def state_adapter():
    return StateAdapter(("a", "b", "c"))


def test_state_str_to_tensor(state_adapter):
    state_str = ("a", "c", "b")
    expected_tensor = torch.tensor([0, 2, 1], dtype=torch.uint8)
    tensor = state_adapter.state_str_to_tensor(state_str)
    assert torch.equal(tensor, expected_tensor)
    assert tensor.shape == (3,)


def test_state_tensor_to_str(state_adapter):
    tensor = torch.tensor([0, 2, 1], dtype=torch.uint8)
    expected_state_str = ("a", "c", "b")
    state_str = state_adapter.state_tensor_to_str(tensor)
    assert state_str == expected_state_str
