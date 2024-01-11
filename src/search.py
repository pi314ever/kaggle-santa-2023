from heapq import heappush, heappop

from utils import PuzzleInstance, PUZZLE_DF, apply_move
from utils.submission import Submission
import multiprocessing as mp
from tqdm import tqdm


def heuristic(state, solution):
    """Calculate the heuristic value of a state."""
    return sum([1 if state[i] != solution[i] else 0 for i in range(len(state))])


def iddfs_search(
    puzzle: PuzzleInstance,
    initial_state: tuple,
    heuristic=heuristic,
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
        steps = dfs_search(puzzle, initial_state, heuristic, depth, verbose=verbose)
        depth += 1
    return steps


def beam_search(
    puzzle: PuzzleInstance,
    initial_state: tuple,
    heuristic=heuristic,
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
            new_state = apply_move(state, puzzle.move_from_str(move))
            new_steps = [*steps, move]
            new_cost = heuristic(new_state, puzzle.solution_state) + len(steps)
            if new_state not in seen:
                heappush(q, (new_cost, new_state, new_steps))
        if len(q) > beam_width:
            q = q[:beam_width]
    print("No solution found")
    return None


def dfs_search(
    puzzle: PuzzleInstance,
    initial_state: tuple,
    heuristic,
    max_steps: int,
    verbose: bool,
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
            [
                (apply_move(state, puzzle.move_from_str(m)), m)
                for m in puzzle.allowed_moves
            ],
            key=lambda x: heuristic(x[0], puzzle.solution_state),
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


def main_mp():
    with Submission() as s, mp.Pool(mp.cpu_count() - 1) as p:
        inputs = []
        ids = []

        for i, row in PUZZLE_DF.iterrows():
            puzzle_type = row["puzzle_type"]
            if puzzle_type not in PUZZLES:
                continue
            id = row["id"]
            initial_state = tuple(row["initial_state"])
            solution_state = tuple(row["solution_state"])
            wildcards = row["num_wildcards"]
            puzzle = PuzzleInstance(puzzle_type, solution_state, wildcards)
            # Add puzzles to queue
            # print(f"Adding puzzle {i} ({puzzle_type})")
            inputs.append(
                (
                    puzzle,
                    initial_state,
                    heuristic,
                    len(s.get_solution(id)),
                    False,
                )
            )
            ids.append(id)
        print("Solving puzzles...")
        results = list(
            tqdm(
                p.imap(lambda x: SEARCH_METHOD(*x), inputs),
                total=len(inputs),
            )
        )
        print("Submitting puzzles...")
        for id, result in zip(ids, results):
            if id == 2:
                print(id, result)
            if result is not None:
                # print(f"Submitting puzzle {id}")
                s.update(id, result, "iddfs")


def main():
    failed = []
    successes = []
    with Submission() as s:
        for i, row in PUZZLE_DF.iterrows():
            puzzle_type = row["puzzle_type"]
            if puzzle_type not in PUZZLES:
                continue
            id = row["id"]
            initial_state = tuple(row["initial_state"])
            solution_state = tuple(row["solution_state"])
            wildcards = row["num_wildcards"]
            puzzle = PuzzleInstance(puzzle_type, solution_state, wildcards)
            print(f"Solving puzzle {i} ({puzzle_type})")
            result = SEARCH_METHOD(
                puzzle,
                initial_state,
                heuristic,
                len(s.get_solution(id)),
                verbose=False,
                beam_width=1000,
            )
            print(result)
            if result is not None:
                print(f"Submitting puzzle {id}")
                success, message = s.update(id, result, SEARCH_METHOD.__name__)
                if not success:
                    print(f"Failed to submit puzzle {id}: {message}")
                    failed.append(id)
                else:
                    print(f"Submitted puzzle {id}")
                    successes.append(id)
    # Summary
    summarize(successes, failed)


def summarize(successes, failed):
    if failed:
        print(f"Failed to submit {len(failed)} puzzles:")
        print(failed)
    print(
        f"Success ratio: {len(failed)} failed | {len(successes)} succeeded = {len(successes) / (len(successes) + len(failed))}"
    )


if __name__ == "__main__":
    print(f"Searching {PUZZLES} with {SEARCH_METHOD.__name__}")
    main()
