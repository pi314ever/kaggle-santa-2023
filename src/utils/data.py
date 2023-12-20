import pandas as pd
from pathlib import Path
import json

from .move import inverse_move

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
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
