import functools as ft
import os
import pathlib
import pickle
import re

import equinox as eqx
import jax
import numpy as np
from jaxtyping import Array, PyTree
from tqdm import tqdm

from statedict2pytree.utils.pydantic_models import ChunkifiedPytreePath, JaxField


def chunkify_pytree(
    tree: PyTree, target_path: str
) -> tuple[ChunkifiedPytreePath, list[JaxField]]:
    """
    Convert a JAX PyTree into chunked files and save them to the specified path.

    Args:
        tree (PyTree): The JAX PyTree to be chunked.
        target_path (str): The directory where chunked files will be saved.

    Returns:
        tuple[ChunkifiedPytreePath, list[JaxField]]: A path to the chunked files
        and a list of JaxFields.

    This function also saves JaxFields as a pickle file in the target directory.
    """
    flattened, _ = jax.tree_util.tree_flatten_with_path(tree)

    for key_path, value in tqdm(flattened):
        key = jax.tree_util.keystr(key_path)
        if not hasattr(value, "shape"):
            continue
        path = pathlib.Path(target_path) / "pytree"

        if not os.path.exists(path):
            os.mkdir(path)

        np.save(path / key, np.array(value))

    jax_fields = pytree_to_fields(tree)
    with open(pathlib.Path(target_path) / "jax_fields.pkl", "wb") as f:
        pickle.dump(jax_fields, f)

    return ChunkifiedPytreePath(path=str(pathlib.Path(target_path))), jax_fields


def serialize_pytree_chunks(tree: PyTree, paths: list[ChunkifiedPytreePath], name: str):
    """
    Reassemble a JAX PyTree from chunked files and serialize it.

    Args:
        tree (PyTree): The original JAX PyTree structure.
        paths (list[ChunkifiedPytreePath]): List of paths to the chunked files.
        name (str): Name of the output serialized file.
    """
    for chunk_path in tqdm(paths):
        array = np.load(chunk_path.path)
        tree = replace_node(tree, chunk_path.path.split(".")[1:-1], array)

    identity = lambda *args, **kwargs: tree
    model, state = eqx.nn.make_with_state(identity)()
    eqx.tree_serialise_leaves(name, (model, state))


def replace_node(tree: PyTree, targets: list[str], new_value: Array) -> PyTree:
    """
    Replace a node in the PyTree with a new value.

    Args:
        tree (PyTree): The PyTree to modify.
        targets (list[str]): Path to the target node.
        new_value (Array): The new value to insert.

    Returns:
        PyTree: The modified PyTree.

    Examples:
    ```python
    import equinox as eqx

    class MyModel(eqx.Module):
        layer1: eqx.nn.Linear
        layer2: eqx.nn.Linear

    model = MyModel(
        layer1=eqx.nn.Linear(10, 20, key=jax.random.key(0)),
        layer2=eqx.nn.Linear(20, 5, key=jax.random.key(1)),
    )
    new_weight = jax.numpy.ones((20, 10))
    updated_model = replace_node(
                        model,
                        ['layer1', 'weight'],
                        new_weight
                    )
    assert (updated_model.layer1.weight == new_weight).all()
    ```
    """
    where = ft.partial(get_node, targets=targets)
    node = where(tree)

    if node is not None and hasattr(node, "shape"):
        tree = eqx.tree_at(
            where,
            tree,
            new_value.reshape(node.shape),
        )
    else:
        print("Couldn't find: ", targets)
    return tree


def get_node(tree: PyTree, targets: list[str]) -> PyTree | None:
    """
    Retrieve a node from the PyTree based on the given path.

    Args:
        tree (PyTree): The PyTree to search.
        targets (list[str]): Path to the target node.

    Returns:
        PyTree | None: The target node if found, None otherwise.

    Examples:
    ```python
    import equinox as eqx

    class MyModel(eqx.Module):
        layer1: eqx.nn.Linear
        layer2: eqx.nn.Linear

    model = MyModel(
            layer1=eqx.nn.Linear(10, 20, key=jax.random.key(0)),
            layer2=eqx.nn.Linear(20, 5, key=jax.random.key(1)),
        )
    layer1_weight = get_node(model, ['layer1', 'weight'])
    assert layer1_weight.shape == (20, 10)
    nonexistent_node = get_node(model, ['layer3'])
    assert nonexistent_node is None
    ```
    """
    if len(targets) == 0 or tree is None:
        return tree
    else:
        next_target: str = targets[0]
        if bool(re.search(r"\[\d+\]", next_target)):
            split_index = next_target.rfind("[")
            name, index = next_target[:split_index], next_target[split_index:]
            index = index[1:-1]
            if hasattr(tree, name):
                subtree = getattr(tree, name)[int(index)]
            else:
                subtree = None
        else:
            if hasattr(tree, next_target):
                subtree = getattr(tree, next_target)
            else:
                subtree = None
        return get_node(subtree, targets[1:])


def pytree_to_fields(pytree: PyTree) -> list[JaxField]:
    """
    Convert a JAX PyTree to a list of JaxField objects.

    Args:
        pytree (PyTree): The JAX PyTree to be converted.

    Returns:
        list[JaxField]: A list of JaxField objects representing the PyTree.
    """
    flattened, _ = jax.tree_util.tree_flatten_with_path(pytree)
    fields = []
    for key_path, value in flattened:
        path = jax.tree_util.keystr(key_path)
        type_path = path.split(".")[1:-1]
        target_path = path.split(".")[1:]
        node_type = type(get_node(pytree, type_path))
        node = get_node(pytree, target_path)
        if node is not None and hasattr(node, "shape") and len(node.shape) > 0:
            fields.append(
                JaxField(path=path, type=str(node_type), shape=tuple(node.shape))
            )

    return fields
