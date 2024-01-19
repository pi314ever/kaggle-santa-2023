from pathlib import Path
import pandas as pd
import re

from .data import DATA_DIR, PUZZLE_DF, PUZZLE_INFO_DF
from .move import apply_move

SUBMISSION_FILE = DATA_DIR / "submission.csv"


def is_valid_solution(id: int, moves: list[str]):
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


class Submission:
    def __init__(self, file: Path = SUBMISSION_FILE) -> None:
        assert file.name.endswith(".csv"), "File must be a .csv file"

        self.file = file
        self.meta_file = file.parent / file.name.replace(".csv", "_meta.csv")
        if self.file.exists():
            self.submission_df = pd.read_csv(self.file)
        else:
            self.submission_df = pd.DataFrame({"id": [], "moves": []})
        self.submission_df.astype({"id": int, "moves": str})

        if self.meta_file.exists():
            self.meta_data = pd.read_csv(self.meta_file)
        else:
            self.meta_data = pd.DataFrame({"id": [], "methods": []})
        self.meta_data.astype({"id": int, "methods": str})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.submit()

    def submit(self):
        self.submission_df.to_csv(self.file, index=False)
        self.meta_data.to_csv(self.meta_file, index=False)

    def update(
        self,
        id: int,
        moves: list[str],
        solution_method: str,
        check_valid_solution=True,
    ) -> tuple[bool, str]:
        if check_valid_solution and not is_valid_solution(id, moves):
            return False, "Invalid solution"
        if id in self.submission_df.index and len(moves) >= len(
            self.submission_df.loc[id]["moves"].split(".")
        ):
            self._add_solution_method(id, solution_method, len(moves))

            return False, "Better solution already exists"

        self.submission_df.loc[id, "id"] = int(id)
        self.submission_df.loc[id, "moves"] = ".".join(moves)
        self._add_solution_method(id, solution_method, len(moves))

        return True, ""

    def _add_solution_method(self, id: int, method: str, solution_length: int):
        entry = f"{method}({solution_length})"
        self.meta_data.loc[id, "id"] = int(id)
        if str(self.meta_data.loc[id]["methods"]) == "nan":
            self.meta_data.loc[id, "methods"] = entry
        else:
            previous_methods = self.meta_data.loc[id]["methods"].split(".")
            matches = [re.match(r"(.*)\(([0-9]+)\)", m) for m in previous_methods]
            lengths = [int(m.group(2)) for m in matches]  # type: ignore
            methods = [m.group(1) for m in matches]  # type: ignore
            # Insert new method
            to_set = []
            for m, l in zip(methods, lengths):
                if method == m:
                    to_set.append(m + f"({min(l, solution_length)})")
                    if solution_length < l:
                        # Need to re-sort
                        to_set = sorted(to_set, key=lambda x: int(x.split("(")[1][:-1]))
                    continue
                to_set.append(m + f"({l})")
            self.meta_data.loc[id, "methods"] = ".".join(to_set)

    def has_solution(self, id: int):
        return id in self.submission_df.index

    def get_solution(self, id: int):
        return self.submission_df.loc[id]["moves"].split(".")
