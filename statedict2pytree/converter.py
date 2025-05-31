import os
import pathlib
import tempfile

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Any, Callable
from jax.tree_util import FlattenedIndexKey, GetAttrKey, KeyPath, SequenceKey
from jaxtyping import Array, PyTree
from pydantic import BaseModel
from tqdm import tqdm


class ChunkifiedPytreePath(BaseModel):
    path: str


class ChunkifiedStatedictPath(BaseModel):
    path: str


class TorchField(BaseModel):
    path: str
    shape: tuple[int, ...]
    skip: bool = False


class JaxField(BaseModel):
    path: KeyPath
    shape: tuple[int, ...]
    skip: bool = False


def is_numerical(element: Any):
    if hasattr(element, "dtype"):
        # Check if it's a JAX or NumPy array
        return (
            np.issubdtype(element.dtype, np.integer)
            or np.issubdtype(element.dtype, np.floating)
            or np.issubdtype(element.dtype, np.complexfloating)
        ) and not np.issubdtype(element.dtype, np.bool_)
    return False


def _default_floating_dtype():
    if jax.config.jax_enable_x64:  # pyright: ignore
        return jnp.float64
    else:
        return jnp.float32


def _can_reshape(shape1: tuple, shape2: tuple):
    """
    Check if two shapes can be reshaped to each other.

    Args:
        shape1 (tuple): First shape.
        shape2 (tuple): Second shape.

    Returns:
        bool: True if shapes can be reshaped to each other, False otherwise.
    """
    product1 = np.prod(shape1)
    product2 = np.prod(shape2)

    return product1 == product2


def _get_stateindex_fields(obj) -> dict:
    state_indices = {}

    for attr_name in dir(obj):
        if attr_name.startswith("_"):
            continue

        try:
            attr_value = getattr(obj, attr_name)
            if isinstance(attr_value, eqx.nn.StateIndex):
                state_indices[attr_name] = attr_value
        except:  # noqa
            pass

    return state_indices


def _get_node(
    tree: PyTree, path: KeyPath, state_indices: dict | None = None
) -> tuple[PyTree | None, dict | None]:
    if tree is None:
        return None, {}
    else:
        if len(path) == 0:
            return tree, state_indices
    f, *_ = path
    if hasattr(tree, "is_stateful"):
        if state_indices is None:
            state_indices = {}
        indices = _get_stateindex_fields(tree)
        for attr_name in indices:
            index: eqx.nn.StateIndex = indices[attr_name]
            assert isinstance(index, eqx.nn.StateIndex)
            state_indices[index.marker] = index
    if isinstance(f, SequenceKey):
        subtree = tree[f.idx]
    elif isinstance(f, GetAttrKey):
        subtree = getattr(tree, f.name)
    elif isinstance(f, FlattenedIndexKey):
        if isinstance(tree, eqx.nn.State):
            assert state_indices is not None
            index = state_indices[f.key]
            subtree = tree.get(index)
        else:
            subtree = None
    else:
        subtree = None
    return _get_node(subtree, path[1:], state_indices)


def _replace_node(
    tree: PyTree, path: KeyPath, new_value: Array, state_indices: dict | None = None
) -> PyTree:
    def where_wrapper(t):
        node, _ = _get_node(t, path=path, state_indices=state_indices)
        return node

    node, _ = _get_node(tree, path=path, state_indices=state_indices)

    if node is not None and eqx.is_array(node):
        tree = eqx.tree_at(
            where_wrapper,
            tree,
            new_value.reshape(node.shape),
        )
    else:
        print("WARNING: Couldn't find: ", jax.tree_util.keystr(path))
    return tree


def move_running_fields_to_the_end(
    torchfields: list[TorchField], identifier: str = "running_"
):
    """
    Helper function to move fields that contain the given string to the end of the
    torchfields. Helpful for stateful layers such as BatchNorm, which appear at the
    end of Equinox pytrees.
    """
    i = 0
    total = 0
    while i + total < len(torchfields):
        if identifier in torchfields[i].path:
            field = torchfields.pop(i)
            torchfields.append(field)
            total += 1
        else:
            i += 1
    return torchfields


def state_dict_to_fields(
    state_dict: dict[str, Any],
) -> list[TorchField]:
    if state_dict is None:
        return []
    fields: list[TorchField] = []
    for key, value in state_dict.items():
        if hasattr(value, "shape") and len(value.shape) > 0:
            fields.append(TorchField(path=key, shape=tuple(value.shape)))
    return fields


def pytree_to_fields(
    pytree: PyTree,
    model_order: list[str] | None = None,
    filter: Callable[[Array], bool] = eqx.is_array,
) -> tuple[list[JaxField], dict | None]:
    jaxfields = []
    paths = jax.tree.leaves_with_path(pytree)
    i = {}
    for p in paths:
        keys, _ = p
        n, i = _get_node(pytree, keys, i)
        if n is not None and filter(n):
            jaxfields.append(JaxField(path=keys, shape=n.shape))

    if model_order is not None:
        ordered_jaxfields = []
        path_dict = {jax.tree_util.keystr(field.path): field for field in jaxfields}

        for path_str in model_order:
            if path_str in path_dict:
                ordered_jaxfields.append(path_dict[path_str])
                del path_dict[path_str]
        ordered_jaxfields.extend(path_dict.values())
        jaxfields = ordered_jaxfields
    return jaxfields, i


def _chunkify_state_dict(
    state_dict: dict[str, np.ndarray], target_path: str
) -> ChunkifiedStatedictPath:
    """
    Convert a PyTorch state dict into chunked files and save them to the specified path.

    Args:
        state_dict (dict[str, np.ndarray]): The PyTorch state dict to be chunked.
        target_path (str): The directory where chunked files will be saved.

    Returns:
        ChunkifiedStatedictPath: A path to the chunked files
    """

    for key in state_dict.keys():
        if not hasattr(state_dict[key], "shape"):
            continue
        path = pathlib.Path(target_path) / "state_dict"

        if not os.path.exists(path):
            os.mkdir(path)
        np.save(path / key, state_dict[key])

    return ChunkifiedStatedictPath(path=str(pathlib.Path(target_path)))


def convert(
    state_dict: dict[str, Any],
    pytree: PyTree,
    jaxfields: list[JaxField],
    state_indices: dict | None,
    torchfields: list[TorchField],
    dtype: Any | None = None,
) -> PyTree:
    if dtype is None:
        dtype = _default_floating_dtype()
    assert dtype is not None
    state_dict_np: dict[str, np.ndarray] = {
        k: state_dict[k].detach().numpy() for k in state_dict
    }

    for k in state_dict_np:
        if np.issubdtype(state_dict_np[k].dtype, np.floating):
            state_dict_np[k] = state_dict_np[k].astype(dtype)

    if len(torchfields) != len(jaxfields):
        raise ValueError(
            f"Length of state_dict ({len(torchfields)}) "
            f"!= length of pytree ({len(jaxfields)})"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        chunkified_statedict_path = _chunkify_state_dict(state_dict_np, tmpdir)
        del state_dict_np, state_dict
        for t, j in tqdm(zip(torchfields, jaxfields), total=len(torchfields)):
            if not _can_reshape(t.shape, j.shape):
                raise ValueError(
                    f"Cannot reshape {t.shape} "
                    f"into shape {j.shape}. "
                    "Note that the order of the fields matters "
                    "and that you can mark arrays as skippable. "
                    f"{t.path=} "
                    f"{jax.tree_util.keystr(j.path)=}"
                )
            state_dict_dir = pathlib.Path(chunkified_statedict_path.path) / "state_dict"
            filename = state_dict_dir / t.path
            new_value = jnp.array(np.load(str(filename) + ".npy"))

            n, _ = _get_node(pytree, j.path, state_indices)
            assert n is not None, f"Node {j.path} not found"
            assert _can_reshape(n.shape, new_value.shape), (
                f"Cannot reshape {n.shape} into {new_value.shape}"
            )

            pytree = _replace_node(pytree, j.path, new_value, state_indices)

    return pytree


def autoconvert(
    pytree: PyTree, state_dict: dict, pytree_model_order: list[str] | None = None
) -> PyTree:
    torchfields = state_dict_to_fields(state_dict)
    jaxfields, state_indices = pytree_to_fields(pytree, pytree_model_order)

    pytree = convert(
        state_dict,
        pytree,
        jaxfields,
        state_indices,
        torchfields,
    )

    return pytree
