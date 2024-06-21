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

from statedict2pytree.utils.pydantic_models import JaxField


def chunkify_pytree(tree: PyTree, target_path: str) -> list[str]:
    paths = []

    flattened, _ = jax.tree_util.tree_flatten_with_path(tree)

    for key_path, value in tqdm(flattened):
        key = jax.tree_util.keystr(key_path)
        if not hasattr(value, "shape"):
            continue
        path = pathlib.Path(target_path) / "pytree"

        if not os.path.exists(path):
            os.mkdir(path)

        np.save(path / key, np.array(value))
        paths.append(str(path / key))

    jax_fields = pytree_to_fields(tree)
    with open(pathlib.Path(target_path) / "jax_fields.pkl", "wb") as f:
        pickle.dump(jax_fields, f)

    return paths


def serialize_pytree_chunks(tree: PyTree, paths: list[str], name: str):
    for path in tqdm(paths):
        array = np.load(path)
        tree = replace_node(tree, path.split(".")[1:-1], array)

    identity = lambda *args, **kwargs: tree
    model, state = eqx.nn.make_with_state(identity)()
    eqx.tree_serialise_leaves(name, (model, state))


def replace_node(tree: PyTree, targets: list[str], new_value: Array) -> PyTree:
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
