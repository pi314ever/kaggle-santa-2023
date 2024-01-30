import re
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import PuzzleWalkDataset
from utils.puzzle_pytorch import PuzzlePyTorch


class ModelDirectoryContext:
    """Manages the files in the directory"""

    def __init__(self, model_dir: Path, config: Optional[dict] = None):
        """Creates a new model directory context. If the model directory does not exist, it is expected to have a config file to set up proper initialization sequence.

        @param model_dir: Path to model directory
        @param config: Config file to use for initialization
        """

        self.model_dir = model_dir

        self.meta_path = model_dir / "meta.yaml"
        self.periodic_model_dir = self.model_dir / "periodic"
        
        self._latest_model: nn.Module
        self._best_model: nn.Module

        if not model_dir.exists():
            # New model directory
            model_dir.mkdir()
            if config is None:
                raise ValueError("Must provide config for new model")
            self.meta = {
                "last_epoch": 0,
                "last_loss": float("inf"),
                "best_epoch": 0,
                "best_loss": float("inf"),
                "loss_history": [],
            }
            self.set_meta_config(config)
            self._latest_model = self.load_initial_model()
            self._best_model = self.load_initial_model()
        else:
            # Existing model directory
            self.meta = yaml.safe_load(open(model_dir / "meta.yaml"))
            if config is not None:
                self.validate_config(config)
            self.load_models()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, _):
        if exc_type is not None:
            print(f"Exception {exc_type} occurred: {exc_value}")
            return False
        self.save()
        return True

    @property
    def is_new(self) -> bool:
        return (
            self.meta["last_epoch"] == 0
            and self.meta["best_epoch"] == 0
            and self.meta["last_loss"] == float("inf")
            and self.meta["best_loss"] == float("inf")
        )

    @property
    def latest_model(self) -> nn.Module:
        return self._latest_model

    @property
    def best_model(self) -> nn.Module:
        return self._best_model

    def set_meta_config(self, config: dict):
        """Sets meta config for a new meta file. Raises ValueError if meta was already set"""
        if not self.is_new:
            print("Meta config is already set. Ignoring.")
            return
        if any(
            k in self.meta for k in ["puzzle_type", "model", "model_args", "train_args"]
        ):
            raise ValueError(
                "Meta config already set. Please check your training configuration."
            )
        self.meta["puzzle_type"] = config["puzzle_type"]
        self.meta["model"] = config["model"]
        self.meta["model_args"] = config["model_args"]
        self.meta["train_args"] = config["train_args"]

    def load_latest_model(self, model: nn.Module):
        """Loads the latest model from a model directory"""
        pattern = re.compile(r"latest_model_(\d+)_(.+).pt")
        latest_model_paths = list(self.model_dir.glob("latest_model_*.pt"))

        if not latest_model_paths:
            raise ValueError(f"No latest model found in {self.model_dir}")

        latest_model_path = max(
            latest_model_paths, key=lambda x: int(pattern.match(x.name).group(1))  # type: ignore
        )
        print("Loading latest model from", latest_model_path)
        model.load_state_dict(torch.load(latest_model_path))
        self._latest_model = model
        return model

    def load_best_model(self, model: nn.Module):
        pattern = re.compile(r"best_model_(\d+)_(.+).pt")
        best_model_paths = list(self.model_dir.glob("best_model_*.pt"))

        if not best_model_paths:
            raise ValueError(f"No best model found in {self.model_dir}")

        best_model_path = max(
            best_model_paths, key=lambda x: float(pattern.match(x.name).group(1))  # type: ignore
        )
        print("Loading best model from", best_model_path)
        model.load_state_dict(torch.load(best_model_path))
        self._best_model = model
        return model

    def load_initial_model(self):
        if "model" not in self.meta or "model_args" not in self.meta:
            raise ValueError(
                "Cannot load initial model without setting meta config. Check meta file consistency."
            )
        return load_model(self.meta["model"], self.meta["model_args"])

    def validate_config(self, config: dict):
        self._validate_meta(config)

    def _validate_meta(self, config: dict):
        errors = []
        if self.meta["puzzle_type"] != puzzle_type:
            errors.append(
                f"Model {args.model_dir} was trained on {self.meta['puzzle_type']}, but config is for {puzzle_type}"
            )
        if self.meta["model"] != config["model"]:
            errors.append(
                f"Model {args.model_dir} was trained with {self.meta['model']}, but config is for {config['model']}"
            )
        for k in config["model_args"]:
            if self.meta["model_args"][k] != config["model_args"][k]:
                errors.append(
                    f"Model {args.model_dir} was trained with {k}={self.meta['model_args'][k]}, but config is for {k}={config['model_args'][k]}"
                )
        ignore_keys = ["num_epochs"]
        for k in config["train_args"]:
            if k in ignore_keys:
                continue
            if self.meta["train_args"][k] != config["train_args"][k]:
                errors.append(
                    f"Model {args.model_dir} was trained with {k}={self.meta['train_args'][k]}, but config is for {k}={config['train_args'][k]}"
                )
        if errors:
            raise ValueError("\n".join(errors))

    def load_models(self):
        model = self.load_initial_model()
        self.load_latest_model(model)
        self.load_best_model(model)

    def update_meta(self, epoch, loss):
        self.meta["last_epoch"] = epoch
        self.meta["last_loss"] = loss
        self.meta["loss_history"].append(loss)
        if loss < self.meta["best_loss"]:
            self.meta["best_loss"] = loss
            self.meta["best_epoch"] = epoch

    def update_model(self, model, epoch, loss):
        self._latest_model = model
        if loss < self.meta["best_loss"]:
            self._best_model = model
        self.update_meta(epoch, loss)

    def save(self):
        torch.save(
            self.latest_model.state_dict(),
            self.model_dir
            / f"latest_model_{self.meta['last_epoch']}_{self.meta['last_loss']}.pt",
        )
        torch.save(
            self.best_model.state_dict(),
            self.model_dir
            / f"best_model_{self.meta['best_epoch']}_{self.meta['best_loss']}.pt",
        )
        self.save_meta()

    def save_meta(self):
        meta_path = self.model_dir / "meta.yaml"
        yaml.safe_dump(self.meta, open(meta_path, "w"))

    def save_model_periodic(self, model: nn.Module, epoch: int, loss: float):
        if not self.periodic_model_dir.exists():
            self.periodic_model_dir.mkdir()

        model_path = self.periodic_model_dir / f"model_{epoch}_{loss}.pt"
        torch.save(model.state_dict(), model_path)


def load_model_from_config(config: dict) -> nn.Module:
    """Loads a model from a config file"""
    return load_model(config["model"], config["model_args"])


def load_model(model_name: str, model_args: dict) -> nn.Module:
    """Loads a model from a config file"""
    if model_name == "resnet":
        from models.resnet import ResnetModel

        model = ResnetModel(**model_args)

    elif model_name == "basic_ff":
        from models.basic_ff import DNNModel

        model = DNNModel(**model_args)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")

    return model


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
    ctx: ModelDirectoryContext,
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
    @param ctx: Model directory context
    @param puzzle_instance: Puzzle instance to train on
    @param num_epochs: Number of epochs to train for
    @param batch_size: Batch size
    @param lr: Learning rate
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
        ctx.update_model(model, epoch, avg_loss)
        if epoch % save_increment == 0:
            ctx.save_model_periodic(model, epoch, avg_loss)
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
    model = load_model_from_config(config)
    solution_state = PUZZLE_DB.get_puzzle_by_type(puzzle_type).iloc[0]["solution_state"]
    puzzle_instance = PuzzlePyTorch(puzzle_type, solution_state, 0)

    dataset = PuzzleWalkDataset(
        puzzle_instance,
        walk_length=config["train_args"]["num_steps"],
        num_walks=config["train_args"]["batch_size"],
    )

    del config["train_args"]["num_steps"]

    with ModelDirectoryContext(args.model_dir, config) as context:
        if context.meta["last_epoch"] == 0 and context.meta["last_loss"] == float(
            "inf"
        ):
            print("Model has not been trained yet. Starting training from scratch")
        else:
            print(
                f"Model has been trained up to epoch {context.meta['last_epoch']} with loss {context.meta['last_loss']}"
            )
            print("Starting training from last checkpoint")

        try:
            train(
                model=context.latest_model,
                ctx=context,
                dataset=dataset,
                start_epoch=context.meta["last_epoch"],
                num_workers=args.num_workers,
                device=args.device,
                **config["train_args"],
            )
        except KeyboardInterrupt:
            print("Training interrupted, saving intermediate model")
