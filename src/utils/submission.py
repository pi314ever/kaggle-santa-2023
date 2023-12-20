from pathlib import Path
import pandas as pd

from .data import DATA_DIR

SUBMISSION_FILE = DATA_DIR / "submission.csv"


def submit_solution(id, moves, submission_file=SUBMISSION_FILE):
    """Submit a solution to the competition."""
    if submission_file.exists():
        submission_df = pd.read_csv(submission_file)
    else:
        submission_df = pd.DataFrame({"id": [], "moves": []})
    submission_df.loc[id, "id"] = id
    submission_df.loc[id, "moves"] = ".".join(moves)
    submission_df.to_csv(submission_file, index=False)
