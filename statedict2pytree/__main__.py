import importlib
import os
import pathlib
import pickle
import tempfile

from beartype.typing import Optional
from jaxtyping import PyTree
from loguru import logger

from statedict2pytree.statedict2pytree import (
    convert_from_path,
)
from statedict2pytree.utils.utils_pytree import chunkify_pytree, serialize_pytree_chunks
from statedict2pytree.utils.utils_state_dict import chunkify_state_dict


def import_module(path: str):
    module_path, attr_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


def convert(
    from_memory: Optional[bool] = False,
    from_path: Optional[bool] = False,
    path_to_state_dict_object: Optional[str] = None,
    path_to_pytree_object: Optional[str] = None,
    chunkify: Optional[bool] = False,
    path_to_statedict_chunks: Optional[str] = None,
    path_to_pytree_chunks: Optional[str] = None,
    arrange_with_anthropic: Optional[bool] = False,
):
    if from_memory:
        pass
    elif from_path:
        if chunkify:
            if path_to_state_dict_object is None or path_to_pytree_object is None:
                raise ValueError(
                    "Provide path_to_state_dict_object and "
                    "path_to_pytree_object to chunkify"
                )
            state_dict: dict = import_module(path_to_state_dict_object)
            with tempfile.TemporaryDirectory() as temp_dir:
                path_to_sd_chunks = (
                    path_to_statedict_chunks if path_to_statedict_chunks else temp_dir
                )
                logger.info(f"Chunkifying state dict at {path_to_sd_chunks}")
                sd_chunks = chunkify_state_dict(state_dict, path_to_sd_chunks)
                logger.info(f"Chunkified state dict saved at {path_to_sd_chunks}")
                logger.info("Deleting state_dict object to free up memory")
                del state_dict
                logger.info("State dict object deleted")
                pytree: PyTree = import_module(path_to_pytree_object)
                path_to_pt_chunks = (
                    path_to_pytree_chunks if path_to_pytree_chunks else temp_dir
                )
                logger.info(f"Chunkifying pytree at {path_to_pt_chunks}")
                pt_chunks = chunkify_pytree(pytree, path_to_pt_chunks)
                logger.info(f"Chunkified pytree saved at {path_to_pt_chunks}")

                with open(
                    str(pathlib.Path(pt_chunks.path) / "jax_fields.pkl"),
                    "rb",
                ) as f:
                    jax_fields = pickle.load(f)

                with open(
                    str(pathlib.Path(sd_chunks.path) / "torch_fields.pkl"),
                    "rb",
                ) as f:
                    torch_fields = pickle.load(f)
            if arrange_with_anthropic:
                logger.info("Arranging chunks with Anthropic")

            convert_from_path(
                jax_fields,
                torch_fields,
                pt_chunks,
                sd_chunks,
            )

            paths = os.listdir(f"{path_to_pt_chunks}/pytree")
            tree_paths = []
            for p in paths:
                tree_paths.append(f"{path_to_pt_chunks}/pytree/" + p)
            logger.info("Serializing pytree chunks")
            serialize_pytree_chunks(pytree, tree_paths, "model.eqx")
            logger.info("Done.")
