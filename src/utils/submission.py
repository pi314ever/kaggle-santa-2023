from pathlib import Path
import pandas as pd

from .data import DATA_DIR
from .move import is_valid_solution

SUBMISSION_FILE = DATA_DIR / "submission.csv"


class Submission:
    def __init__(self, file: Path = SUBMISSION_FILE) -> None:
        self.file = file
        if self.file.exists():
            self.submission_df = pd.read_csv(self.file)
        else:
            self.submission_df = pd.DataFrame({"id": [], "moves": []})
        self.submission_df.astype({"id": int, "moves": str})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.submit()

    def submit(self):
        self.submission_df.to_csv(self.file, index=False)

    def update(self, id: int, moves: list[str], check_valid_solution=True):
        if id in self.submission_df.index and len(moves) >= len(
            self.submission_df.loc[id]["moves"].split(".")
        ):
            # Better solution already exists
            return
        if check_valid_solution and not is_valid_solution(id, moves):
            print(f"Invalid solution for id {id}")
            return

        self.submission_df.loc[id, "id"] = int(id)
        self.submission_df.loc[id, "moves"] = ".".join(moves)

    def has_solution(self, id: int):
        return id in self.submission_df.index

    def get_solution(self, id: int):
        return self.submission_df.loc[id]["moves"].split(".")


if __name__ == "__main__":
    with Submission() as s:
        s.update(0, ["F", "B", "U"])
        s.update(1, ["F", "B", "U"])
        s.update(2, ["F", "B", "U"])
        print(s.submission_df.head())
