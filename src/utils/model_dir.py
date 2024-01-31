import re
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from .load_model import load_model


class ModelDirectoryManager:
    """Manages the model directory structure and updating. Assumes directory is already created and populated.

    Model directory structure:

    <model_dir>
    --- periodic
    --- --- model_<epoch>_<loss>.pt
    --- --- ...
    --- meta.json
    --- latest_model_<epoch>_<loss>.pt
    --- best_model_<epoch>_<loss>.pt

    """

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir

        self.meta_path = model_dir / "meta.yaml"
        self.periodic_dir = model_dir / "periodic"

        assert self.model_dir.exists(), f"Model directory not found at {self.model_dir}"
        assert (
            self.periodic_dir.exists()
        ), f"Periodic directory not found at {self.periodic_dir}"
        assert self.meta_path.exists(), f"Meta file not found at {self.meta_path}"

        self.meta = yaml.safe_load(open(self.meta_path, "r"))

        self._latest_model = self.load_latest_model(self.load_initial_model())
        self._best_model = self.load_best_model(self.load_initial_model())

    @property
    def latest_model(self):
        return self._latest_model

    @property
    def best_model(self):
        return self._best_model

    @staticmethod
    def create_from_config(model_dir: Path, config: dict) -> "ModelDirectoryManager":
        """Create a new model directory with initial model and meta files"""
        assert not model_dir.exists(), f"Model directory already exists at {model_dir}"

        model_dir.mkdir(parents=True)
        (model_dir / "periodic").mkdir()

        # Create initial meta file
        meta = {
            "epoch": 0,
            "best_epoch": 0,
            "best_loss": float("inf"),
            "best_loss_epoch": 0,
            "config": config,
        }
        meta["puzzle_type"] = config["puzzle_type"]
        meta["model"] = config["model"]
        meta["model_args"] = config["model_args"]
        meta["train_args"] = config["train_args"]

        yaml.safe_dump(meta, open(model_dir / "meta.yaml", "w"))

        # Create initial model
        model = load_model(config["model"], config["model_args"])

        torch.save(model.state_dict(), model_dir / "latest_model_0_inf.pt")
        torch.save(model.state_dict(), model_dir / "best_model_0_inf.pt")

        return ModelDirectoryManager(model_dir)

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

    def validate_with_config(self, config: dict):
        self._validate_meta(config)

    def _validate_meta(self, config: dict):
        errors = []
        if self.meta["puzzle_type"] != config["puzzle_type"]:
            errors.append(
                f"Model {self.model_dir} was trained on {self.meta['puzzle_type']}, but config is for {config['puzzle_type']}"
            )
        if self.meta["model"] != config["model"]:
            errors.append(
                f"Model {self.model_dir} was trained with {self.meta['model']}, but config is for {config['model']}"
            )
        for k in config["model_args"]:
            if self.meta["model_args"][k] != config["model_args"][k]:
                errors.append(
                    f"Model {self.model_dir} was trained with {k}={self.meta['model_args'][k]}, but config is for {k}={config['model_args'][k]}"
                )
        ignore_keys = ["num_epochs"]
        for k in config["train_args"]:
            if k in ignore_keys:
                continue
            if self.meta["train_args"][k] != config["train_args"][k]:
                errors.append(
                    f"Model {self.model_dir} was trained with {k}={self.meta['train_args'][k]}, but config is for {k}={config['train_args'][k]}"
                )
        if errors:
            raise ValueError("\n".join(errors))

    def update_meta(self, epoch: int, loss: float):
        self.meta["last_epoch"] = epoch
        self.meta["last_loss"] = loss
        self.meta["loss_history"].append(loss)
        if loss < self.meta["best_loss"]:
            self.meta["best_loss"] = loss
            self.meta["best_epoch"] = epoch

    def update_model(self, model: nn.Module, epoch: int, loss: float):
        self._latest_model = model
        if loss < self.meta["best_loss"]:
            self._best_model = model
        self.update_meta(epoch, loss)

    def save(self):
        torch.save(
            self._latest_model.state_dict(),
            self.model_dir
            / f"latest_model_{self.meta['last_epoch']}_{self.meta['last_loss']}.pt",
        )
        torch.save(
            self._best_model.state_dict(),
            self.model_dir
            / f"best_model_{self.meta['best_epoch']}_{self.meta['best_loss']}.pt",
        )
        self.save_meta()

    def save_meta(self):
        meta_path = self.model_dir / "meta.yaml"
        yaml.safe_dump(self.meta, open(meta_path, "w"))

    def save_model_periodic(self, model: nn.Module, epoch: int, loss: float):
        model_path = self.periodic_dir / f"model_{epoch}_{loss}.pt"
        torch.save(model.state_dict(), model_path)
