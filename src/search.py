from heapq import heappush, heappop

from utils import Puzzle, PUZZLE_DF, apply_move, submit_solution, ROOT_DIR


def heuristic(state, solution):
    """Calculate the heuristic value of a state."""
    return sum([1 if state[i] != solution[i] else 0 for i in range(len(state))])


def search(
    puzzle: Puzzle,
    heuristic=heuristic,
    wildcards: int = 0,
    max_steps: int = 1000,
    verbose: bool = False,
):
    """Basic search algorithm with heuristic function."""
    q = []
    seen = set()
    heappush(q, (0, puzzle.initial_state, []))
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
    raise ValueError("No solution found.")


def main():
    for i, row in PUZZLE_DF.iterrows():
        puzzle_type = row["puzzle_type"]
        initial_state = tuple(row["initial_state"])
        solution_state = tuple(row["solution_state"])
        puzzle = Puzzle(puzzle_type, initial_state, solution_state)
        wildcards = row["num_wildcards"]
        steps = search(puzzle, wildcards=wildcards, verbose=False)
        submit_solution(row["id"], steps, ROOT_DIR / "a_star_submission.csv")
        print(f"Submitted solution for puzzle {i} ({len(steps)}): {steps}")


if __name__ == "__main__":
    main()
