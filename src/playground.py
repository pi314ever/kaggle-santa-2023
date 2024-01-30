# File to explore model

from pathlib import Path

from train import ModelDirectoryContext
from utils.data import PUZZLE_DB
from utils.puzzle_pytorch import PuzzlePyTorch


class Explorer:
    def __init__(self, model_dir: Path, puzzle_instance: PuzzlePyTorch):
        self.model_dir = model_dir
        self.puzzle = puzzle_instance
        self.ctx = ModelDirectoryContext(model_dir)

    def explore(self):
        "Infinite loop text input to explore model"
        state = self.puzzle.solution_state
        model = self.ctx.latest_model
        model.train(False)
        allowed_moves = set(self.puzzle.allowed_moves)
        moves = []
        while True:
            print(f"Current state: {state}")
            print(
                f"Cost: {float(model.forward(self.puzzle.state_adapter.state_str_to_tensor(state)))}"
            )
            print(f"Possible actions: {self.puzzle.allowed_moves}")
            action = input("Enter action: ")
            if action == "history":
                print(moves)
                continue
            if action == "reset":
                state = self.puzzle.solution_state
                moves = []
                print("\n\nReset\n\n")
                continue
            if action == "undo":
                if len(moves) == 0:
                    print("No moves to undo")
                    continue
                state = self.puzzle.move_state(state, self.puzzle.moves.opposite_move_str(moves.pop()))
                continue
            if action == "exit":
                break
            if action not in allowed_moves:
                print(f"Invalid action {action}")
                continue
            moves.append(action)
            state = self.puzzle.move_state(state, action)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("puzzle", type=str, help="Puzzle to solve, e.g. cube_2/2/2")
    parser.add_argument("model_dir", type=Path, help="Path to model directory")

    args = parser.parse_args()

    solution_state = PUZZLE_DB.get_puzzle_by_type(args.puzzle).iloc[0]["solution_state"]
    puzzle = PuzzlePyTorch(args.puzzle, solution_state)

    explorer = Explorer(args.model_dir, puzzle)
    explorer.explore()
