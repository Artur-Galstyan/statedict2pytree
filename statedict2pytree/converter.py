import functools as ft
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
from tqdm import tqdm

from statedict2pytree.utils.pydantic_models import (
    ChunkifiedPytreePath,
    ChunkifiedStatedictPath,
    JaxField,
    TorchField,
)
from statedict2pytree.utils.utils import can_reshape, match_using_anthropic
from statedict2pytree.utils.utils_pytree import (
    chunkify_pytree,
    get_node,
    pytree_to_fields,
    serialize_pytree_chunks,
)
from statedict2pytree.utils.utils_state_dict import (
    chunkify_state_dict,
    pad_with_skip_layers,
    state_dict_to_fields,
)


def import_module(path: str):
    """
    Import a module and return a specific attribute from it.

    Args:
        path (str): The full path to the module and attribute, separated by a dot.

    Returns:
        The specified attribute from the imported module.
    """
    module_path, attr_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


def autoconvert_state_dict_to_pytree(
    pytree: PyTree, state_dict: dict
) -> tuple[PyTree, eqx.nn.State]:
    """
    Automatically convert a PyTorch state dict to a JAX pytree.

    Args:
        pytree (PyTree): The target JAX pytree structure.
        state_dict (dict): The source PyTorch state dict.

    Returns:
        tuple: A tuple containing the converted pytree and its associated state.
    """
    jax_fields = pytree_to_fields(pytree)
    torch_fields = state_dict_to_fields(state_dict)

    for k, v in state_dict.items():
        state_dict[k] = v.numpy()
    return convert_from_pytree_and_state_dict(
        jax_fields, torch_fields, pytree, state_dict
    )


def autoconvert_from_paths(
    chunkified_pytree_path: ChunkifiedPytreePath,
    chunkified_state_dict_path: ChunkifiedStatedictPath,
):
    """
    Automatically convert PyTorch state dict to JAX pytree using file paths.

    To "make" the paths, use
    `statedict2pytree.utils.utils_state_dict.chunkify_state_dict`.
    `statedict2pytree.utils.utils_pytree.chunkify_pytree`.

    Args:
        chunkified_pytree_path (ChunkifiedPytreePath): Path to the JAX pytree
        pickle file.
        chunkified_state_dict_path (ChunkifiedStatedictPath): Path to the
        PyTorch state dict pickle file.
    """
    with open(
        str(pathlib.Path(chunkified_pytree_path.path) / "jax_fields.pkl"), "rb"
    ) as f:
        jax_fields = pickle.load(f)

    with open(
        str(pathlib.Path(chunkified_state_dict_path.path) / "torch_fields.pkl"), "rb"
    ) as f:
        torch_fields = pickle.load(f)
    convert_from_path(
        jax_fields, torch_fields, chunkified_pytree_path, chunkified_state_dict_path
    )


def convert_from_pytree_and_state_dict(
    jax_fields: list[JaxField],
    torch_fields: list[TorchField],
    pytree: PyTree,
    state_dict: dict[str, np.ndarray],
) -> tuple[PyTree, eqx.nn.State]:
    """
    Convert PyTorch state dict to JAX pytree using
    in-memory structures and field mappings.

    Args:
        jax_fields (list[JaxField]): List of JAX fields.
        torch_fields (list[TorchField]): List of PyTorch fields.
        pytree (PyTree): The target JAX pytree structure.
        state_dict (dict[str, np.ndarray]): The source PyTorch state dict.

    Returns:
        tuple: A tuple containing the converted pytree and its associated state.

    Raises:
        ValueError: If fields have incompatible shapes.
    """
    identity = lambda *args, **kwargs: pytree
    model, state = eqx.nn.make_with_state(identity)()
    state_paths: list[tuple[JaxField, TorchField]] = []
    for i in range(len(jax_fields)):
        torch_field = torch_fields[i]
        jax_field = jax_fields[i]
        if torch_field.skip:
            continue
        if not can_reshape(jax_field.shape, torch_field.shape):
            raise ValueError(
                "Fields have incompatible shapes! "
                f"{jax_field.shape=} != {torch_field.shape=}"
            )
        path = jax_field.path.split(".")[1:]
        if "StateIndex" in jax_field.type:
            state_paths.append((jax_field, torch_field))

        else:
            where = ft.partial(get_node, targets=path)
            if where(model) is not None:
                model = eqx.tree_at(
                    where,
                    model,
                    state_dict[torch_field.path].reshape(jax_field.shape),
                )
    result: dict[str, list[TorchField]] = {}
    for tuple_item in state_paths:
        path_prefix = tuple_item[0].path.split(".")[1:-1]
        prefix_key = ".".join(path_prefix)
        if prefix_key not in result:
            result[prefix_key] = []
        result[prefix_key].append(tuple_item[1])

    for key in result:
        state_index = get_node(model, key.split("."))
        if state_index is not None:
            to_replace_tuple = tuple([state_dict[i.path] for i in result[key]])
            state = state.set(state_index, to_replace_tuple)
    return model, state


def convert_from_path(
    jax_fields: list[JaxField],
    torch_fields: list[TorchField],
    chunkified_pytree_path: ChunkifiedPytreePath,
    chunkified_statedict_path: ChunkifiedStatedictPath,
):
    """
    Convert PyTorch state dict to JAX pytree using file paths and field mappings.

    Args:
        jax_fields (list[JaxField]): List of JAX fields.
        torch_fields (list[TorchField]): List of PyTorch fields.
        chunkified_pytree_path (ChunkifiedPytreePath): Path to the
        JAX pytree directory.
        chunkified_statedict_path (ChunkifiedStatedictPath): Path to the
        PyTorch state dict directory.

    Raises:
        ValueError: If fields have incompatible shapes.
    """
    j_path = pathlib.Path(chunkified_pytree_path.path)
    t_path = pathlib.Path(chunkified_statedict_path.path)

    for jax_field, torch_field in tqdm(zip(jax_fields, torch_fields)):
        if torch_field.skip:
            continue
        if not can_reshape(jax_field.shape, torch_field.shape):
            raise ValueError(
                "Fields have incompatible shapes!"
                f"{jax_field.shape=} != {torch_field.shape=}"
            )
        pt_path = pathlib.Path(j_path) / "pytree" / jax_field.path
        sd_path = pathlib.Path(t_path) / "state_dict" / torch_field.path

        if pt_path.exists():
            os.remove(pt_path)
        np.save(pt_path, np.load(str(sd_path) + ".npy"))


def arrange(torch_fields, jax_fields, anthropic_model):
    """
    Arrange torch_fields to match jax_fields using Anthropic's language model.

    Args:
        torch_fields: List of PyTorch fields.
        jax_fields: List of JAX fields.
        anthropic_model: The Anthropic model to use for matching.

    Returns:
        List of arranged torch_fields.

    Raises:
        ValueError: If failed to match using Anthropic or
        if ANTHROPIC_API_KEY is not set.
    """
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
    chunkify: bool = False,
    path_to_state_dict_chunks: Optional[str] = None,
    path_to_pytree_chunks: Optional[str] = None,
    arrange_with_anthropic: bool = False,
    anthropic_model: Literal["haiku", "opus", "sonnet", "sonnet3.5"] = "sonnet3.5",
    target_name: str = "model.eqx",
    target_dir: str = ".",
):
    """
    Convert a PyTorch state dict to a JAX pytree, with various
    options for input and processing.

    Args:
        from_memory (bool): If True, convert using in-memory objects.
        from_path (bool): If True, convert using file paths.
        state_dict (dict): PyTorch state dict (required if
        from_memory is True).
        pytree (PyTree): JAX pytree (required if from_memory is True).
        path_to_state_dict_object (str): Path to state dict object
        (required if from_path is True).
        path_to_pytree_object (str): Path to pytree object
        (required if from_path is True).
        chunkify (bool): If True, chunkify the state dict and pytree.
        path_to_state_dict_chunks (str): Path to save state dict chunks.
        path_to_pytree_chunks (str): Path to save pytree chunks.
        arrange_with_anthropic (bool): If True, use Anthropic to arrange fields.
        anthropic_model (str): Anthropic model to use for arrangement.
        target_name (str): Name of the output file.
        target_dir (str): Directory to save the output file.

    Raises:
        ValueError: If neither from_memory nor from_path is True, or if
        required arguments are missing.
    """
    if not from_memory and not from_path:
        raise ValueError("Provide either from_memory or from_path")
    logger.info("Starting conversion...")
    if from_memory:
        if state_dict is None:
            raise ValueError("state_dict object must not be None")
        _convert_from_memory(
            state_dict,
            pytree,
            arrange_with_anthropic,
            anthropic_model,
            target_dir,
            target_name,
        )
    elif from_path:
        _convert_from_path(
            path_to_state_dict_chunks,
            path_to_state_dict_object,
            chunkify,
            path_to_pytree_chunks,
            path_to_pytree_object,
            arrange_with_anthropic,
            anthropic_model,
            target_dir,
            target_name,
        )


def _convert_from_memory(
    state_dict: dict[str, torch.Tensor],
    pytree: PyTree,
    arrange_with_anthropic: bool,
    anthropic_model: str,
    target_dir: str,
    target_name: str,
):
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


def _convert_from_path(
    path_to_state_dict_chunks: Optional[str],
    path_to_state_dict_object: Optional[str],
    chunkify: bool,
    path_to_pytree_chunks: Optional[str],
    path_to_pytree_object: Optional[str],
    arrange_with_anthropic: bool,
    anthropic_model: str,
    target_dir: str,
    target_name: str,
):
    with tempfile.TemporaryDirectory() as temp_dir:
        path_to_sd_chunks = (
            path_to_state_dict_chunks if path_to_state_dict_chunks else temp_dir
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
            path_to_torch_fields = pathlib.Path(path_to_sd_chunks) / "torch_fields.pkl"
            with open(path_to_torch_fields, "rb") as f:
                torch_fields = pickle.load(f)
            sd_chunks = ChunkifiedStatedictPath(path=path_to_sd_chunks)

        path_to_pt_chunks = path_to_pytree_chunks if path_to_pytree_chunks else temp_dir

        imported_pytree: Optional[PyTree] = None
        if path_to_pytree_object is None:
            raise ValueError("Provide path_to_pytree_object")
        imported_pytree = import_module(path_to_pytree_object)
        if chunkify:
            logger.info(f"Chunkifying pytree at {path_to_pt_chunks}")
            pt_chunks, jax_fields = chunkify_pytree(imported_pytree, path_to_pt_chunks)
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
