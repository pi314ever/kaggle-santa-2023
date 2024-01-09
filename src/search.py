from heapq import heappush, heappop

from utils import PuzzleOld, PUZZLE_DF, apply_move, submit_solution, ROOT_DIR
from utils.submission import Submission
import multiprocessing as mp
from tqdm import tqdm


def heuristic(state, solution):
    """Calculate the heuristic value of a state."""
    return sum([1 if state[i] != solution[i] else 0 for i in range(len(state))])


def search(
    puzzle: PuzzleOld,
    initial_state: tuple,
    heuristic=heuristic,
    wildcards: int = 0,
    max_steps: int = 40,
    verbose: bool = False,
):
    """Basic search algorithm with heuristic function."""
    q = []
    seen = set()
    heappush(q, (0, initial_state, []))
    while q:
        cost, state, steps = heappop(q)
        if verbose:
            print(f"Cost: {cost}, Steps: {steps}, State: {state}")
        if puzzle.is_approximately_solved(state, wildcards):
            if verbose:
                print("Solution found")
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
            heappush(q, (new_cost, new_state, new_steps))
        if len(q) > 500:
            q = q[:500]
    raise ValueError("No solution found.")


def failsafe_search(
    puzzle: PuzzleOld,
    initial_state: tuple,
    heuristic=heuristic,
    wildcards: int = 0,
    max_steps: int = 40,
    verbose: bool = False,
):
    try:
        return search(puzzle, initial_state, heuristic, wildcards, max_steps, verbose)
    except:
        return None


def imap_search(args):
    return failsafe_search(*args)


def main():
    max_steps = 5000
    SKIP_SEEN = False
    with Submission() as s, mp.Pool(mp.cpu_count() - 1) as p:
        inputs = []
        ids = []

        for i, row in PUZZLE_DF.iterrows():
            puzzle_type = row["puzzle_type"]
            if puzzle_type not in [
                "cube_2/2/2",
                "wreath_6/6",
                "wreath_7/7",
                "wreath_12/12",
            ]:
                continue
            initial_state = tuple(row["initial_state"])
            solution_state = tuple(row["solution_state"])
            puzzle = PuzzleOld(puzzle_type, solution_state)
            wildcards = row["num_wildcards"]
            if s.has_solution(row["id"]):
                if SKIP_SEEN:
                    print(f"Skipping puzzle {i} ({puzzle_type})")
                    continue
            # Add puzzles to queue
            print(f"Adding puzzle {i} ({puzzle_type})")
            inputs.append(
                (puzzle, initial_state, heuristic, wildcards, max_steps, False)
            )
            ids.append(row["id"])
        print("Solving puzzles...")
        results = list(
            tqdm(
                p.imap_unordered(imap_search, inputs),
                total=len(inputs),
            )
        )
        print("Submitting puzzles...")
        for id, result in zip(ids, results):
            if result is not None:
                print(f"Submitting puzzle {id}")
                s.update(id, result)


if __name__ == "__main__":
    main()
