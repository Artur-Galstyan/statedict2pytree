import importlib
import os
import pathlib
import pickle
import tempfile

import beartype
import equinox as eqx
import numpy as np
import torch
from beartype.typing import Literal, Optional
from dotenv import load_dotenv
from jaxtyping import PyTree
from loguru import logger

from statedict2pytree.statedict2pytree import (
    convert_from_path,
    convert_from_pytree_and_state_dict,
)
from statedict2pytree.utils.pydantic_models import (
    ChunkifiedPytreePath,
    ChunkifiedStatedictPath,
)
from statedict2pytree.utils.utils import match_using_anthropic
from statedict2pytree.utils.utils_pytree import (
    chunkify_pytree,
    pytree_to_fields,
    serialize_pytree_chunks,
)
from statedict2pytree.utils.utils_state_dict import (
    chunkify_state_dict,
    pad_with_skip_layers,
    state_dict_to_fields,
)


def import_module(path: str):
    module_path, attr_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


def arrange(torch_fields, jax_fields, anthropic_model):
    logger.info("Arranging chunks with Anthropic")
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY", None)
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set.")
    torch_fields = pad_with_skip_layers(
        torch_fields, abs(len(jax_fields) - len(torch_fields))
    )

    val = "n"
    while val != "y":
        torch_fields = match_using_anthropic(
            jax_fields, torch_fields, anthropic_model, api_key
        )

        logger.info("We got the following match:")
        if not torch_fields:
            raise ValueError("Failed to match using Anthropic!")
        for jf, tf in zip(jax_fields, torch_fields):
            log = logger.info if jf.shape == tf.shape else logger.warning
            log(f"{jf.path} ({jf.shape}) -> {tf.path} ({tf.shape})")
        val = input("Is this match correct? (y/n): ").strip().lower()
        if val == "y":
            break
        elif val == "n":
            continue
        else:
            logger.warning(f"{val=}, invalid input")
            continue


@beartype.beartype
def convert(
    from_memory: Optional[bool] = False,
    from_path: Optional[bool] = False,
    state_dict: Optional[dict[str, torch.Tensor]] = None,
    pytree: Optional[PyTree] = None,
    path_to_state_dict_object: Optional[str] = None,
    path_to_pytree_object: Optional[str] = None,
    chunkify: Optional[bool] = False,
    path_to_statedict_chunks: Optional[str] = None,
    path_to_pytree_chunks: Optional[str] = None,
    arrange_with_anthropic: Optional[bool] = False,
    anthropic_model: Literal["haiku", "opus", "sonnet", "sonnet3.5"] = "sonnet3.5",
    target_name: str = "model.eqx",
    target_dir: str = ".",
):
    if not from_memory and not from_path:
        raise ValueError("Provide either from_memory or from_path")
    logger.info("Starting conversion...")
    if from_memory:
        if state_dict is None or pytree is None:
            raise ValueError(
                "Provide path_to_state_dict_object and "
                "path_to_pytree_object to chunkify"
            )
        state_dict_np: dict[str, np.ndarray] = dict()
        for k in state_dict:
            state_dict_np[k] = state_dict[k].numpy()

        torch_fields = state_dict_to_fields(state_dict_np)
        jax_fields = pytree_to_fields(pytree)

        if len(torch_fields) != len(jax_fields):
            raise ValueError("state_dict and pytree have different lengths!")

        if arrange_with_anthropic:
            arrange(torch_fields, jax_fields, anthropic_model)
        model, state = convert_from_pytree_and_state_dict(
            jax_fields, torch_fields, pytree, state_dict_np
        )
        eqx.tree_serialise_leaves(target_dir + "/" + target_name, (model, state))
        logger.info("Done.")
    elif from_path:
        with tempfile.TemporaryDirectory() as temp_dir:
            path_to_sd_chunks = (
                path_to_statedict_chunks if path_to_statedict_chunks else temp_dir
            )
            if chunkify:
                imported_state_dict: Optional[dict[str, np.ndarray]] = None
                if path_to_state_dict_object is None:
                    raise ValueError("Provide path_to_state_dict_object to chunkify")
                imported_state_dict = import_module(path_to_state_dict_object)
                logger.info(f"Chunkifying state dict at {path_to_sd_chunks}")
                assert imported_state_dict is not None
                sd_chunks, torch_fields = chunkify_state_dict(
                    imported_state_dict, path_to_sd_chunks
                )
                logger.info(f"Chunkified state dict saved at {path_to_sd_chunks}")
                logger.info("Deleting state_dict object to free up memory")
                del imported_state_dict
                logger.info("State dict object deleted")
            else:
                path_to_torch_fields = (
                    pathlib.Path(path_to_sd_chunks) / "torch_fields.pkl"
                )
                with open(path_to_torch_fields, "rb") as f:
                    torch_fields = pickle.load(f)
                sd_chunks = ChunkifiedStatedictPath(path=path_to_sd_chunks)

            path_to_pt_chunks = (
                path_to_pytree_chunks if path_to_pytree_chunks else temp_dir
            )

            imported_pytree: Optional[PyTree] = None
            if path_to_pytree_object is None:
                raise ValueError("Provide path_to_pytree_object")
            imported_pytree = import_module(path_to_pytree_object)
            if chunkify:
                logger.info(f"Chunkifying pytree at {path_to_pt_chunks}")
                pt_chunks, jax_fields = chunkify_pytree(
                    imported_pytree, path_to_pt_chunks
                )
                logger.info(f"Chunkified pytree saved at {path_to_pt_chunks}")
            else:
                path_to_jax_fields = pathlib.Path(path_to_pt_chunks) / "jax_fields.pkl"
                with open(path_to_jax_fields, "rb") as f:
                    jax_fields = pickle.load(f)
                pt_chunks = ChunkifiedPytreePath(path=path_to_pt_chunks)
            if arrange_with_anthropic:
                arrange(torch_fields, jax_fields, anthropic_model)

            convert_from_path(
                jax_fields,
                torch_fields,
                pt_chunks,
                sd_chunks,
            )

            paths = os.listdir(f"{path_to_pt_chunks}/pytree")
            tree_paths: list[ChunkifiedPytreePath] = []
            for p in paths:
                tree_paths.append(
                    ChunkifiedPytreePath(path=f"{path_to_pt_chunks}/pytree/" + p)
                )
            logger.info("Serializing pytree chunks")
            serialize_pytree_chunks(
                imported_pytree, tree_paths, target_dir + "/" + target_name
            )
            logger.info("Done.")
