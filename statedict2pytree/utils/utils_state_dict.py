import os
import pathlib
import pickle

import numpy as np
from beartype.typing import Optional
from tqdm import tqdm

from statedict2pytree.utils.pydantic_models import TorchField


def chunkify_state_dict(
    state_dict: dict[str, np.ndarray], target_path: str
) -> list[str]:
    paths = []

    for key in tqdm(state_dict.keys()):
        if not hasattr(state_dict[key], "shape"):
            continue
        path = pathlib.Path(target_path) / "state_dict"

        if not os.path.exists(path):
            os.mkdir(path)

        np.save(path / key, state_dict[key])
        paths.append(str(path / key))

    torch_fields = state_dict_to_fields(state_dict)
    with open(pathlib.Path(target_path) / "torch_fields.pkl", "wb") as f:
        pickle.dump(torch_fields, f)
    return paths


def state_dict_to_fields(state_dict: Optional[dict]) -> list[TorchField]:
    if state_dict is None:
        return []
    fields: list[TorchField] = []
    for key, value in state_dict.items():
        if hasattr(value, "shape") and len(value.shape) > 0:
            fields.append(TorchField(path=key, shape=tuple(value.shape)))
    return fields
