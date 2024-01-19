import torch

from .typing import STATE_TYPE


class StateAdapter:
    "Converts a tuple[str, ...] state into a tensor of integer indices"

    def __init__(self, choices: tuple[str, ...]):
        self.choices = choices
        self.num_choices = len(choices)
        self.choices_to_idx = {choice: i for i, choice in enumerate(choices)}

    def state_str_to_tensor(
        self, state_str: STATE_TYPE, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """Returns a integer-encoded tensor of the state of shape len(state_str)"""
        state_idx = [self.choices_to_idx[choice] for choice in state_str]
        return torch.tensor(state_idx, device=device, dtype=torch.uint8).unsqueeze(0)

    def state_tensor_to_str(self, state: torch.Tensor) -> STATE_TYPE:
        """Returns a string representation of the state"""
        assert (
            state.shape[0] == self.num_choices
        ), f"State shape is {state.shape}, but the number of choices is {self.num_choices}"
        assert (
            state.dtype == torch.uint8
        ), f"State dtype is {state.dtype}, should be torch.uint8"
        return tuple(self.choices[int(i)] for i in state)
