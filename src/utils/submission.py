from pathlib import Path
import pandas as pd

from .data import DATA_DIR

SUBMISSION_FILE = DATA_DIR / "submission.csv"


class Submission:
    def __init__(self, file: Path = SUBMISSION_FILE) -> None:
        self.file = file
        if self.file.exists():
            self.submission_df = pd.read_csv(self.file)
        else:
            self.submission_df = pd.DataFrame({"id": [], "moves": []})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.submit()

    def submit(self):
        self.submission_df.to_csv(self.file, index=False)

    def update(self, id: int, moves: list[str]):
        # Take best solution by move length
        if id - 1 in self.submission_df.index and len(moves) >= len(
            self.submission_df.loc[id - 1]["moves"].split(".")
        ):
            return
        self.submission_df.loc[id - 1, "id"] = int(id)
        self.submission_df.loc[id - 1, "moves"] = ".".join(moves)

    def has_solution(self, id: int):
        return id - 1 in self.submission_df.index

    def get_solution(self, id: int):
        return self.submission_df.loc[id - 1]["moves"].split(".")
