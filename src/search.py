import argparse
from heapq import heappop, heappush
from pathlib import Path
from typing import Callable

from tqdm import tqdm

from utils.data import PUZZLE_DB, PUZZLE_DF, PUZZLE_INFO_DF
from utils.puzzle import Puzzle
from utils.puzzle_pytorch import PuzzlePyTorch
from utils.state import StateAdapter
from utils.submission import Submission
from utils.typing import STATE_TYPE


def model_heuristic_builder(model, state_adapter: StateAdapter, device):
    """Meta function that builds a heuristic function given a model"""

    def heuristic(state, solution_state, steps):
        """Heuristic function that uses a model to predict the number of steps remaining"""

        return float(model.forward(state_adapter.state_str_to_tensor(state, device=device)))

    return heuristic


def dist_heuristic(state, solution_state, steps):
    """Basic heuristic function that calculates distance from solution state by the different characters"""
    return sum(int(s != t) for s, t in zip(state, solution_state)) + len(steps)


def iddfs_search(
    puzzle: Puzzle,
    initial_state: tuple,
    heuristic: Callable[[STATE_TYPE, STATE_TYPE, list], float],
    max_steps: int = 40,
    verbose: bool = False,
    **kwargs,
):
    """Basic search algorithm with heuristic function using iddfs."""
    depth = 1
    steps = None

    while depth <= max_steps and steps is None:
        if verbose:
            print(f"Searching with depth {depth}")
        steps = dfs_search(
            puzzle, initial_state, depth, heuristic=heuristic, verbose=verbose
        )
        depth += 1
    return steps


def beam_search(
    puzzle: Puzzle,
    initial_state: tuple,
    heuristic: Callable[[STATE_TYPE, STATE_TYPE, list], float],
    max_steps: int = 40,
    verbose: bool = False,
    beam_width: int = 500,
    **kwargs,
):
    q = []
    seen = set()
    heappush(q, (0, initial_state, []))
    while q:
        cost, state, steps = heappop(q)
        if verbose:
            print(f"Cost: {cost}, Steps: {steps}, State: {state}")
        if puzzle.is_solved(state):
            if verbose:
                print(f"Solution found ({len(steps)}): {steps}")
            return steps
        if state in seen:
            continue
        seen.add(tuple(state))
        if len(steps) > max_steps:
            continue
        for move in puzzle.allowed_moves:
            new_state = puzzle.move_state(state, move)
            new_steps = [*steps, move]
            new_cost = heuristic(new_state, puzzle.solution_state, new_steps)
            if new_state not in seen:
                heappush(q, (new_cost, new_state, new_steps))
        if len(q) > beam_width:
            q = q[:beam_width]
    print("No solution found")
    return None


def dfs_search(
    puzzle: Puzzle,
    initial_state: tuple,
    max_steps: int,
    verbose: bool,
    heuristic: Callable[[STATE_TYPE, STATE_TYPE, list], float],
    **kwargs,
):
    """Uses DFS to search for a valid solution within max steps"""
    steps = []
    seen = set()

    def visit(state, move=None):
        if state in seen:
            return
        seen.add(tuple(state))
        if len(steps) > max_steps:
            return
        if move is not None:
            steps.append(move)
        if puzzle.is_solved(state):
            if verbose:
                print("Solution found")
            return steps
        # Order search of new state by heuristic
        new_states_and_moves = sorted(
            [(puzzle.move_state(state, m), m) for m in puzzle.allowed_moves],
            key=lambda x: heuristic(x[0], puzzle.solution_state, steps),
        )
        for new_state, m in new_states_and_moves:
            res = visit(new_state, m)
            if res is not None:
                return res
        if move is not None:
            steps.pop()

    return visit(initial_state)


PUZZLES = [
    "cube_2/2/2",
    "wreath_6/6",
    "wreath_7/7",
    "wreath_12/12",
]

SEARCH_METHOD = beam_search


def summarize(successes, failed):
    if failed:
        print(f"Failed to submit {len(failed)} puzzles:")
        print(failed)
    print(
        f"Success ratio: {len(failed)} failed | {len(successes)} succeeded = {len(successes) / (len(successes) + len(failed))}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "puzzle",
        type=str,
        help="Puzzle to solve, e.g. cube_2/2/2",
        choices=list(PUZZLE_INFO_DF["puzzle_type"].unique()),
    )
    parser.add_argument(
        "--method",
        type=str,
        default="beam_search",
        help="Search method to use",
        choices=["beam_search", "iddfs_search"],
    )
    parser.add_argument(
        "--pytorch_model_dir",
        type=Path,
        default=None,
        help="Path to pytorch model. Defaults to using standard heuristic search if none is provided.",
    )
    parser.add_argument("--max_steps", type=int, default=40, help="Max steps to search")
    parser.add_argument(
        "--beam_width",
        type=int,
        default=500,
        help="Beam width search parameter for beam search only",
    )

    parser.add_argument("--device", type=str, default="cpu", help="Device to use for the PyTorch model" )

    args = parser.parse_args()

    if args.pytorch_model_dir is not None:
        print("Using pytorch model")
        PUZZLE_CLASS = PuzzlePyTorch
        # TODO: load model
        from train import ModelDirectoryContext
        from utils.state import StateAdapter

        # State adapter might need to be loaded in a different place.
        state_adapter = StateAdapter(PUZZLE_DB.state_choices(args.puzzle)[0])

        ctx = ModelDirectoryContext(args.pytorch_model_dir)
        model = ctx.latest_model
        model.train(False)
        model.to(args.device)

        heuristic = model_heuristic_builder(model, state_adapter, args.device)
    else:
        PUZZLE_CLASS = Puzzle
        heuristic = dist_heuristic

    SEARCH_METHOD = globals()[args.method]

    failed = []
    successes = []

    with Submission() as s:
        for i, row in PUZZLE_DB.get_puzzle_by_type(args.puzzle).iterrows():
            puzzle_instance = PUZZLE_CLASS(
                args.puzzle, tuple(row["solution_state"]), int(row["num_wildcards"])
            )
            id = int(row["id"])
            initial_state = tuple(row["initial_state"])
            print(f"Solving puzzle {id} ({args.puzzle})")
            result = SEARCH_METHOD(
                puzzle_instance,
                initial_state,
                max_steps=args.max_steps,
                heuristic=heuristic,
                verbose=True,
                beam_width=args.beam_width,
            )
            if result is not None:
                print(f"Submitting puzzle {id}")
                success, message = s.update(id, result, SEARCH_METHOD.__name__)
                if not success:
                    print(f"Failed to submit puzzle {id}: {message}")
                    failed.append(id)
                else:
                    print(f"Submitted puzzle {id}")
                    successes.append(id)

    summarize(successes, failed)
