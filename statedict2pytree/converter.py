import importlib
import os
import tempfile

import equinox as eqx
from beartype.typing import Literal, Optional
from dotenv import load_dotenv
from jaxtyping import PyTree
from loguru import logger

from statedict2pytree.statedict2pytree import (
    convert_from_path,
    convert_from_pytree_and_state_dict,
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


def convert(
    from_memory: Optional[bool] = False,
    from_path: Optional[bool] = False,
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
    logger.info("Starting conversion...")
    if from_memory:
        if path_to_state_dict_object is None or path_to_pytree_object is None:
            raise ValueError(
                "Provide path_to_state_dict_object and "
                "path_to_pytree_object to chunkify"
            )

        state_dict: dict = import_module(path_to_state_dict_object)
        torch_fields = state_dict_to_fields(state_dict)

        pytree: PyTree = import_module(path_to_pytree_object)
        jax_fields = pytree_to_fields(pytree)

        if arrange_with_anthropic:
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
                    logger.info(f"{jf.path} ({jf.shape}) -> {tf.path} ({tf.shape})")
                val = input("Is this match correct? (y/n): ").strip().lower()
                if val == "y":
                    break
                elif val == "n":
                    continue
                else:
                    logger.warning(f"{val=}, invalid input")
                    continue

        model, state = convert_from_pytree_and_state_dict(
            jax_fields, torch_fields, pytree, state_dict
        )
        eqx.tree_serialise_leaves(target_dir + "/" + target_name, (model, state))
        logger.info("Done.")
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
                sd_chunks, torch_fields = chunkify_state_dict(
                    state_dict, path_to_sd_chunks
                )
                logger.info(f"Chunkified state dict saved at {path_to_sd_chunks}")
                logger.info("Deleting state_dict object to free up memory")
                del state_dict
                logger.info("State dict object deleted")
                pytree: PyTree = import_module(path_to_pytree_object)
                path_to_pt_chunks = (
                    path_to_pytree_chunks if path_to_pytree_chunks else temp_dir
                )
                logger.info(f"Chunkifying pytree at {path_to_pt_chunks}")
                pt_chunks, jax_fields = chunkify_pytree(pytree, path_to_pt_chunks)
                logger.info(f"Chunkified pytree saved at {path_to_pt_chunks}")

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
            serialize_pytree_chunks(pytree, tree_paths, target_dir + "/" + target_name)
            logger.info("Done.")
