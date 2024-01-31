from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.functional import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import PuzzleWalkDataset
from utils.load_model import load_model
from utils.model_dir import ModelDirectoryManager
from utils.puzzle_pytorch import PuzzlePyTorch


class ModelDirectoryContext:
    """Manages the files in the directory"""

    def __init__(self, model_dir: Path, config: dict):
        """Creates a new model directory context. If the model directory does not exist, it is expected to have a config file to set up proper initialization sequence.

        @param model_dir: Path to model directory
        @param config: Config file to use for initialization
        """

        if model_dir.exists():
            self.model_manager = ModelDirectoryManager(model_dir)
        else:
            self.model_manager = ModelDirectoryManager.create_from_config(
                model_dir, config
            )

    def __enter__(self):
        return self.model_manager

    def __exit__(self, exc_type, exc_value, _):
        if exc_type is not None:
            print(f"Exception {exc_type} occurred: {exc_value}")
            return False
        self.model_manager.save()
        return True


def validate_config(config: dict):
    errors = []
    if "puzzle_type" not in config:
        errors.append("Missing puzzle type (key: puzzle_type)")
    if "model" not in config:
        errors.append("Missing model name (key: model)")
    if "model_args" not in config:
        errors.append("Missing model arguments (key: model_args)")
    if "train_args" not in config:
        errors.append("Missing training arguments (key: train_args)")
    else:
        training_args = [
            "num_epochs",
            "batch_size",
            "learning_rate",
            "num_steps",
            "batch_size",
        ]
        for k in training_args:
            if k not in config["train_args"]:
                errors.append(f"Missing training argument {k} (key: train_args.{k})")
    if errors:
        raise ValueError("\n".join(errors))


def train(
    model: nn.Module,
    dataset: PuzzleWalkDataset,
    mgr: ModelDirectoryManager,
    start_epoch: int,
    num_epochs: int,
    batch_size: int,
    save_increment: int,
    num_workers: int,
    learning_rate: float,
    device: torch.device,
):
    """Train a model on a puzzle instance

    @param model: Model to train
    @param dataset: Dataset to train on
    @param mgr: Model directory manager to save model to
    @param start_epoch: Epoch to start training from
    @param num_epochs: Number of total epochs to train for
    @param batch_size: Batch size
    @param save_increment: Number of epochs between saving model periodically
    @param num_workers: Number of workers to use for data loading
    @param learning_rate: Learning rate
    @param device: Device to train on
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    loss_fn = nn.L1Loss()
    print(f"Starting training {dataset.puzzle.puzzle_type} on {device}.")
    for epoch in tqdm(range(start_epoch + 1, num_epochs + 1)):
        total_loss = 0
        # Train on solution state
        optimizer.zero_grad()
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        num_samples = 0
        for states, y in dataloader:
            states = states.reshape(-1, states.shape[2])
            states = states.to(device)
            y = y.reshape(-1, 1)
            y = y.to(device)
            pred = model.forward(states)
            loss = loss_fn(pred, y)
            total_loss += float(loss)
            loss.backward()
            optimizer.step()
            num_samples += states.shape[0]
        avg_loss = total_loss / num_samples
        tqdm.write(f"Epoch {epoch}: loss {avg_loss}")
        mgr.update_model(model, epoch, avg_loss)
        if epoch % save_increment == 0:
            mgr.save_model_periodic(model, epoch, avg_loss)
    return model


if __name__ == "__main__":
    import argparse

    from utils.data import PUZZLE_DB

    argparser = argparse.ArgumentParser()
    argparser.add_argument("config_file", type=Path)
    argparser.add_argument("model_dir", type=Path, default="models")
    argparser.add_argument("--device", type=str, default="cpu")
    argparser.add_argument("--num_workers", type=int, default=0)

    args = argparser.parse_args()

    args.device = torch.device(args.device)

    config = yaml.safe_load(open(args.config_file))
    validate_config(config)
    puzzle_type = config["puzzle_type"]

    print(f"Loading config from {args.config_file}: puzzle type {puzzle_type}")
    model = load_model(config["model"], config["model_args"])
    solution_state = PUZZLE_DB.get_puzzle_by_type(puzzle_type).iloc[0]["solution_state"]
    puzzle_instance = PuzzlePyTorch(puzzle_type, solution_state, 0)

    dataset = PuzzleWalkDataset(
        puzzle_instance,
        walk_length=config["train_args"]["num_steps"],
        num_walks=config["train_args"]["batch_size"],
    )

    # Prepare config for keyward expansion in train
    del config["train_args"]["num_steps"]

    with ModelDirectoryContext(args.model_dir, config) as manager:
        if manager.meta["last_epoch"] == 0 and manager.meta["last_loss"] == float(
            "inf"
        ):
            print("Model has not been trained yet. Starting training from scratch")
        else:
            print(
                f"Model has been trained up to epoch {manager.meta['last_epoch']} with loss {manager.meta['last_loss']}"
            )
            print("Starting training from last checkpoint")

        try:
            train(
                model=manager.latest_model,
                mgr=manager,
                dataset=dataset,
                start_epoch=manager.meta["last_epoch"],
                num_workers=args.num_workers,
                device=args.device,
                **config["train_args"],
            )
        except KeyboardInterrupt:
            print("Training interrupted, saving intermediate model")
