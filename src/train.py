from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from utils.puzzle_pytorch import PuzzlePyTorch


class ModelDirectoryContext:
    def __init__(self, model_dir: Path, config: dict):
        self.model_dir = model_dir
        self.meta, is_new = load_meta(self.model_dir)
        if is_new:
            self.meta["puzzle_type"] = config["puzzle_type"]
            self.meta["model"] = config["model"]
            self.meta["model_args"] = config["model_args"]
            self.meta["train_args"] = config["train_args"]
        else:
            self._validate_meta(config)
        self.model = self.load_model(config)
        self.best_model = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, _):
        self.save()
        if exc_type is not None:
            print(f"Exception {exc_type} occurred: {exc_value}")
            return False
        return True

    def _validate_meta(self, config: dict):
        errors = []
        if meta["puzzle_type"] != puzzle_type:
            errors.append(
                f"Model {args.model_dir} was trained on {meta['puzzle_type']}, but config is for {puzzle_type}"
            )
        if meta["model"] != config["model"]:
            errors.append(
                f"Model {args.model_dir} was trained with {meta['model']}, but config is for {config['model']}"
            )
        for k in config["model_args"]:
            if meta["model_args"][k] != config["model_args"][k]:
                errors.append(
                    f"Model {args.model_dir} was trained with {k}={meta['model_args'][k]}, but config is for {k}={config['model_args'][k]}"
                )
        ignore_keys = ["num_epochs"]
        for k in config["train_args"]:
            if k in ignore_keys:
                continue
            if meta["train_args"][k] != config["train_args"][k]:
                errors.append(
                    f"Model {args.model_dir} was trained with {k}={meta['train_args'][k]}, but config is for {k}={config['train_args'][k]}"
                )

        if errors:
            raise ValueError("\n".join(errors))

    def load_model(self, config: dict):
        model = load_model_from_config(config)
        model, _, _ = load_model_incremental(model, self.model_dir, torch.device("cpu"))
        return model

    def update_meta(self, epoch, loss):
        self.meta["last_epoch"] = epoch
        self.meta["last_loss"] = loss
        self.meta["loss_history"].append(loss)
        if loss < self.meta["best_loss"]:
            self.meta["best_loss"] = loss
            self.meta["best_epoch"] = epoch

    def update_model(self, model, epoch, loss):
        self.model = model
        if loss < self.meta["best_loss"]:
            self.best_model = model
        self.update_meta(epoch, loss)

    def save(self):
        torch.save(self.model.state_dict(), self.model_dir / "latest_model.pt")
        if self.best_model is not None:
            torch.save(
                self.best_model.state_dict(),
                self.model_dir
                / f"best_model_{self.meta['best_epoch']}_{self.meta['best_loss']}.pt",
            )
        self.save_meta()

    def save_meta(self):
        meta_path = self.model_dir / "meta.yaml"
        yaml.safe_dump(self.meta, open(meta_path, "w"))


def load_meta(model_dir: Path):
    meta_path = model_dir / "meta.yaml"
    if meta_path.exists():
        meta = yaml.safe_load(open(meta_path))
        is_new = False
    else:
        meta = {
            "last_epoch": 0,
            "last_loss": float("inf"),
            "best_epoch": 0,
            "best_loss": float("inf"),
            "loss_history": [],
        }
        is_new = True
    return meta, is_new


def load_model_incremental(model: nn.Module, model_dir: Path, device: torch.device):
    """Load a model from a model directory

    @param model: Model to load
    @param path: Path to load from
    @param device: Device to load on

    @return: Loaded model
    @return: Last epoch it was trained on (0 if not trained)
    @return: Last loss it was trained on (inf if not trained)
    """
    if not model_dir.is_dir():
        raise ValueError(f"{model_dir} is not a directory")

    if not model_dir.exists():
        model_dir.mkdir()

    latest_model_path = model_dir / "latest_model.pt"
    meta, _ = load_meta(model_dir)

    if latest_model_path.exists():
        model.load_state_dict(torch.load(latest_model_path, map_location=device))
    return model, meta["last_epoch"], meta["last_loss"]


def save_model_periodic(model: nn.Module, model_dir: Path, epoch: int, loss: float):
    periodic_model_path = model_dir / "periodic" / f"model_{epoch}_{loss}.pt"
    if not periodic_model_path.parent.exists():
        periodic_model_path.parent.mkdir()

    if periodic_model_path.exists():
        periodic_model_path.unlink()

    torch.save(model.state_dict(), periodic_model_path)


def load_model_from_config(config: dict) -> nn.Module:
    """Loads a model from a config file"""
    model_name = config["model"]
    if model_name == "resnet":
        from models.resnet import ResnetModel

        model = ResnetModel(**config["model_args"])

    elif model_name == "basic_ff":
        from models.basic_ff import DNNModel

        model = DNNModel(**config["model_args"])
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")

    return model


def train(
    model: nn.Module,
    ctx: ModelDirectoryContext,
    puzzle_instance: PuzzlePyTorch,
    start_epoch: int,
    num_epochs: int,
    batch_size: int,
    num_steps: int,
    save_increment: int,
    save_dir: Path,
    lr: float,
    device: torch.device,
):
    """Train a model on a puzzle instance

    @param model: Model to train
    @param ctx: Model directory context
    @param puzzle_instance: Puzzle instance to train on
    @param num_epochs: Number of epochs to train for @param batch_size: Batch size
    @param batch_size: Batch size
    @param num_steps: Number of steps to take in a random walk
    @param lr: Learning rate
    @param device: Device to train on
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    loss_fn = nn.L1Loss()
    print(f"Starting training {puzzle_instance.puzzle_type} on {device}.")
    for epoch in tqdm(range(start_epoch, num_epochs + 1)):
        total_loss = 0
        # Train on solution state
        optimizer.zero_grad()
        states = puzzle_instance.solution_state_tensor.repeat(batch_size, 1)
        states = states.to(device)
        pred = model.forward(states)
        loss = loss_fn(pred, torch.zeros_like(pred, device=device))
        total_loss += float(loss)
        loss.backward()
        optimizer.step()

        for step, (states, _) in enumerate(
            puzzle_instance.random_walk_tensor(states, num_steps)
        ):
            optimizer.zero_grad()
            pred = model.forward(states)
            loss = loss_fn(pred, (step + 1) * torch.ones_like(pred, device=device))
            total_loss += float(loss)
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / (num_steps + 1) / batch_size
        tqdm.write(f"Epoch {epoch}: loss {avg_loss}")
        ctx.update_model(model, epoch, avg_loss)
        if epoch % save_increment == 0:
            save_model_periodic(model, save_dir, epoch, avg_loss)
    return model


if __name__ == "__main__":
    import argparse

    from utils.data import PUZZLE_DB

    argparser = argparse.ArgumentParser()
    argparser.add_argument("config_file", type=Path)
    argparser.add_argument("model_dir", type=Path, default="models")
    argparser.add_argument("--device", type=str, default="cpu")

    args = argparser.parse_args()

    args.device = torch.device(args.device)

    if not args.model_dir.exists():
        args.model_dir.mkdir()

    config = yaml.safe_load(open(args.config_file))
    puzzle_type = config["puzzle_type"]

    print(f"Loading config from {args.config_file}: puzzle type {puzzle_type}")
    model = load_model_from_config(config)
    solution_state = PUZZLE_DB.get_puzzle_by_type(puzzle_type).iloc[0]["solution_state"]
    puzzle_instance = PuzzlePyTorch(puzzle_type, solution_state, 0)

    if not args.model_dir.exists():
        args.model_dir.mkdir()

    meta, is_new = load_meta(args.model_dir)

    # Check that the  model loaded is the same as the one in the config

    model, last_epoch, last_loss = load_model_incremental(
        model, args.model_dir, args.device
    )

    if last_epoch == 0 and last_loss == float("inf"):
        print("Model has not been trained yet. Starting training from scratch")
    else:
        print(f"Model has been trained up to epoch {last_epoch} with loss {last_loss}")
    with ModelDirectoryContext(args.model_dir, config) as context:
        try:
            train(
                model=model,
                ctx=context,
                puzzle_instance=puzzle_instance,
                start_epoch=last_epoch,
                num_epochs=config["train_args"]["num_epochs"],
                batch_size=config["train_args"]["batch_size"],
                num_steps=config["train_args"]["num_steps"],
                lr=config["train_args"]["learning_rate"],
                save_increment=config["train_args"]["save_increment"],
                save_dir=args.model_dir,
                device=args.device,
            )
        except KeyboardInterrupt:
            print("Training interrupted, saving intermediate model")
