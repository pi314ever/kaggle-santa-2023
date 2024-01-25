import torch

from .typing import STATE_TYPE


class StateAdapter:
    "Converts a tuple[str, ...] state into a tensor of integer indices"

    def __init__(self, choices: tuple[str, ...]):
        self.choices = choices
        self.num_choices = len(choices)
        self.choices_to_idx = {choice: i for i, choice in enumerate(choices)}

    def state_str_to_tensor(
        self,
        state_str: STATE_TYPE | list[STATE_TYPE],
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Returns a integer-encoded tensor of the state of shape len(state_str)"""
        if not isinstance(state_str, list):
            state_str = [state_str]
        state_idx = [
            [self.choices_to_idx[choice] for choice in state] for state in state_str
        ]
        return torch.tensor(state_idx, device=device, dtype=torch.uint8).reshape(
            (len(state_str), -1)
        )

    def state_tensor_to_str(self, state: torch.Tensor) -> STATE_TYPE:
        """Returns a string representation of the state"""
        assert (
            torch.max(state) < self.num_choices
            and torch.unique(state).shape[0] == self.num_choices
        ), f"State must have num_choices={self.num_choices} unique elements. Got {torch.unique(state).shape[0]} unique elements"
        assert (
            state.dtype == torch.uint8
        ), f"State dtype is {state.dtype}, should be torch.uint8"

        assert state.ndim == 1, f"State must be 1-dimensional. Got {state.ndim}"

        return tuple(self.choices[int(i)] for i in state)

    def batch_state_tensor_to_str(self, states: torch.Tensor) -> list[STATE_TYPE]:
        assert (
            torch.max(states) < self.num_choices
            and torch.unique(states).shape[0] == self.num_choices
        ), f"States must have num_choices={self.num_choices} unique elements. Got {torch.unique(states).shape[0]} unique elements"
        assert (
            states.dtype == torch.uint8
        ), f"State dtype is {states.dtype}, should be torch.uint8"
        return [
            tuple(self.choices[int(i)] for i in states[j])
            for j in range(states.shape[0])
        ]
