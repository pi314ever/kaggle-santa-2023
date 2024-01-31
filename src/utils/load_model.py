from models.resnet import ResnetModel
from models.distnn import DistNN

def load_model(model_name: str, model_args: dict):
    """Loads an untrained model from arguments and model name"""

    if model_name == "resnet":
        model = ResnetModel(**model_args)
    elif model_name == "distnn":
        model = DistNN(**model_args)
    else:
        raise ValueError(f"Unrecognized model name {model_name}")

    return model
