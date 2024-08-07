import os
import pathlib
import pickle

import numpy as np
from beartype.typing import Optional
from tqdm import tqdm

from statedict2pytree.utils.pydantic_models import ChunkifiedStatedictPath, TorchField


def chunkify_state_dict(
    state_dict: dict[str, np.ndarray], target_path: str
) -> tuple[ChunkifiedStatedictPath, list[TorchField]]:
    """
    Convert a PyTorch state dict into chunked files and save them to the specified path.

    Args:
        state_dict (dict[str, np.ndarray]): The PyTorch state dict to be chunked.
        target_path (str): The directory where chunked files will be saved.

    Returns:
        tuple[ChunkifiedStatedictPath, list[TorchField]]: A path to the chunked files
        and a list of TorchFields.


    This function also saves TorchFields as a pickle file in the target directory.
    """
    paths = []

    for key in tqdm(state_dict.keys()):
        if not hasattr(state_dict[key], "shape"):
            continue
        path = pathlib.Path(target_path) / "state_dict"

        if not os.path.exists(path):
            os.mkdir(path)
        np.save(path / key, state_dict[key])

    torch_fields = state_dict_to_fields(state_dict)
    with open(pathlib.Path(target_path) / "torch_fields.pkl", "wb") as f:
        pickle.dump(torch_fields, f)
    return ChunkifiedStatedictPath(path=str(pathlib.Path(target_path))), torch_fields


def state_dict_to_fields(state_dict: Optional[dict]) -> list[TorchField]:
    """
    Convert a PyTorch state dict to a list of TorchField objects.

    Args:
        state_dict (Optional[dict]): The PyTorch state dict to be converted.

    Returns:
        list[TorchField]: A list of TorchField objects representing the state dict.
    """
    if state_dict is None:
        return []
    fields: list[TorchField] = []
    for key, value in state_dict.items():
        if hasattr(value, "shape") and len(value.shape) > 0:
            fields.append(TorchField(path=key, shape=tuple(value.shape)))
    return fields


def pad_with_skip_layers(fields: list[TorchField], n: int) -> list[TorchField]:
    for _ in range(n):
        fields.append(TorchField(path="SKIP", shape=(), skip=True))
    return fields
