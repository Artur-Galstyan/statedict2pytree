import os
import sys
import tempfile
from types import ModuleType

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import torch
from jaxtyping import Array
from statedict2pytree.converter import convert
from statedict2pytree.utils.utils_pytree import chunkify_pytree
from statedict2pytree.utils.utils_state_dict import chunkify_state_dict


def test_conversion_from_memory():
    name = "model.eqx"
    from_memory = True
    lin1_shape = 10, 10
    lin2_shape = 20, 20

    class JModel(eqx.Module):
        lin1: Array
        lin2: Array

        def __init__(self):
            self.lin1 = jnp.zeros(shape=lin1_shape)
            self.lin2 = jnp.zeros(shape=lin2_shape)

    pytree = JModel()
    state_dict = {
        "lin1": torch.ones(size=lin1_shape),
        "lin2": torch.ones(size=lin2_shape),
    }

    convert(
        from_memory=from_memory, state_dict=state_dict, pytree=pytree, target_name=name
    )

    pytree = eqx.tree_deserialise_leaves(name, like=pytree)
    assert jnp.allclose(pytree.lin1, jnp.ones(shape=lin1_shape))
    assert jnp.allclose(pytree.lin2, jnp.ones(shape=lin2_shape))

    os.remove(name)


def test_invalid_sizes():
    name = "model.eqx"
    from_memory = True
    lin1_shape = 10, 10
    lin2_shape = 20, 20

    class JModel(eqx.Module):
        lin1: Array
        lin2: Array

        def __init__(self):
            self.lin1 = jnp.zeros(shape=lin1_shape)
            self.lin2 = jnp.zeros(shape=lin2_shape)

    pytree = JModel()
    state_dict = {
        "lin1": torch.ones(size=lin1_shape),
        "lin2": torch.ones(size=lin2_shape),
        "lin3": torch.ones(size=lin2_shape),
    }

    exception_thrown = False
    try:
        convert(
            from_memory=from_memory,
            state_dict=state_dict,
            pytree=pytree,
            target_name=name,
        )
    except ValueError:
        exception_thrown = True

    assert exception_thrown


def test_conversion_from_path_with_chunks():
    name = "model.eqx"
    from_path = True
    shape = 10, 10
    arrays = 10

    class JModel(eqx.Module):
        arrays: list[Array]

        def __init__(self):
            self.arrays = [jnp.zeros(shape=shape) for _ in range(arrays)]

    pytree = JModel()
    state_dict = {f"arrays.[{i}]": torch.ones(size=shape) for i in range(arrays)}

    temp_module = ModuleType("temp_module")
    temp_module.pytree = pytree  # pyright: ignore
    temp_module.state_dict = state_dict  # pyright: ignore
    sys.modules["temp_module"] = temp_module

    try:
        convert(
            path_to_state_dict_object="temp_module.state_dict",
            path_to_pytree_object="temp_module.pytree",
            target_name=name,
            from_path=from_path,
            chunkify=True,
        )
    finally:
        # Clean up
        del sys.modules["temp_module"]

    pytree = eqx.tree_deserialise_leaves(name, like=pytree)
    for array in pytree.arrays:
        assert jnp.allclose(array, jnp.ones(shape))

    os.remove(name)


def test_conversion_from_path_without_chunks():
    name = "model.eqx"
    from_path = True
    shape = 10, 10
    arrays = 10

    class JModel(eqx.Module):
        arrays: list[Array]

        def __init__(self):
            self.arrays = [jnp.zeros(shape=shape) for _ in range(arrays)]

    pytree = JModel()
    state_dict = {f"arrays.[{i}]": np.ones(shape=shape) for i in range(arrays)}

    temp_module = ModuleType("temp_module")
    temp_module.pytree = pytree  # pyright: ignore
    temp_module.state_dict = state_dict  # pyright: ignore
    sys.modules["temp_module"] = temp_module

    with tempfile.TemporaryDirectory() as tempdir:
        sd_chunks, torch_fields = chunkify_state_dict(state_dict, tempdir)
        pt_chunks, jax_fields = chunkify_pytree(pytree, tempdir)

        convert(
            path_to_pytree_object="temp_module.pytree",
            path_to_pytree_chunks=pt_chunks.path,
            path_to_statedict_chunks=sd_chunks.path,
            target_name=name,
            from_path=from_path,
            chunkify=False,
        )

        pytree = eqx.tree_deserialise_leaves(name, like=pytree)
        for array in pytree.arrays:
            assert jnp.allclose(array, jnp.ones(shape))

        os.remove(name)

    del sys.modules["temp_module"]
