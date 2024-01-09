import pandas as pd
from pathlib import Path
import json

from .move import inverse_move

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
PUZZLE_INFO_DF = pd.read_csv(DATA_DIR / "puzzle_info.csv")
PUZZLE_DF = pd.read_csv(DATA_DIR / "puzzles.csv")

PUZZLE_DF["initial_state"] = PUZZLE_DF["initial_state"].apply(
    lambda x: tuple(x.split(";"))
)
PUZZLE_DF["solution_state"] = PUZZLE_DF["solution_state"].apply(
    lambda x: tuple(x.split(";"))
)

# Converting the string representation of allowed_moves to dictionary
PUZZLE_INFO_DF["allowed_moves"] = PUZZLE_INFO_DF["allowed_moves"].apply(
    lambda x: json.loads(x.replace("'", '"'))
)


# Add inverse moves to the allowed moves
new_allowed_moves = {}
for puzzle in PUZZLE_INFO_DF["puzzle_type"]:
    new_allowed_moves[puzzle] = {}
    allowed_moves = PUZZLE_INFO_DF[PUZZLE_INFO_DF["puzzle_type"] == puzzle][
        "allowed_moves"
    ].iloc[0]
    for k, move in allowed_moves.items():
        new_allowed_moves[puzzle][k] = move
        new_allowed_moves[puzzle]["-" + k] = inverse_move(move)

PUZZLE_INFO_DF["allowed_moves"] = PUZZLE_INFO_DF["puzzle_type"].apply(
    lambda x: new_allowed_moves[x]
)


def get_solution_states(puzzle_type):
    return set(PUZZLE_DF[PUZZLE_DF["puzzle_type"] == puzzle_type]["solution_state"])


# PUZZLE_INFO_DF["solution_states"] = PUZZLE_INFO_DF["puzzle_type"].apply(
#     lambda x: get_solution_states(x)
# )


# Wrapper around the puzzle database to make it easier to use
class PuzzleDB:
    def __init__(self) -> None:
        self.puzzle_df = PUZZLE_DF
        self.puzzle_info_df = PUZZLE_INFO_DF

    def allowed_moves(self, puzzle_type: str) -> dict:
        """Returns a dict of label -> move permutations for a particular puzzle type"""
        return self.puzzle_info_df[self.puzzle_info_df["puzzle_type"] == puzzle_type][
            "allowed_moves"
        ].iloc[0]

    def iter_puzzles(self, filter=None):
        if filter is None:
            return self.puzzle_df.iterrows()
        else:
            return self.puzzle_df[filter].iterrows()

    def get_puzzle_by_type(self, puzzle_type: str):
        return self.puzzle_df[self.puzzle_df["puzzle_type"] == puzzle_type]

    def get_puzzle_by_id(self, id: int):
        return self.puzzle_df[self.puzzle_df["id"] == id]


PUZZLE_DB = PuzzleDB()
