import numpy as np

from statedict2pytree.utils.pydantic_models import JaxField, TorchField


def can_reshape(shape1: tuple, shape2: tuple):
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


def field_jsons_to_fields(
    jax_fields_json, torch_fields_json
) -> tuple[list[JaxField], list[TorchField]]:
    """
    Convert JSON representations of JAX and PyTorch fields to
    JaxField and TorchField objects.

    Args:
        jax_fields_json: JSON representation of JAX fields.
        torch_fields_json: JSON representation of PyTorch fields.

    Returns:
        tuple[list[JaxField], list[TorchField]]: A tuple containing lists of
        JaxField and TorchField objects.
    """
    jax_fields: list[JaxField] = []
    for f in jax_fields_json:
        jax_fields.append(
            JaxField(path=f["path"], type=f["type"], shape=tuple(f["shape"]))
        )

    torch_fields: list[TorchField] = []
    for f in torch_fields_json:
        torch_fields.append(
            TorchField(path=f["path"], shape=tuple(f["shape"]), skip=f["skip"])
        )
    return jax_fields, torch_fields
