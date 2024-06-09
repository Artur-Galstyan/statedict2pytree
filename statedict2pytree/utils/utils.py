import numpy as np

from statedict2pytree.utils.pydantic_models import JaxField, TorchField


def can_reshape(shape1, shape2):
    product1 = np.prod(shape1)
    product2 = np.prod(shape2)

    return product1 == product2


def field_jsons_to_fields(
    jax_fields_json, torch_fields_json
) -> tuple[list[JaxField], list[TorchField]]:
    jax_fields: list[JaxField] = []
    for f in jax_fields_json:
        shape_tuple = tuple(
            [int(i) for i in f["shape"].strip("()").split(",") if len(i) > 0]
        )
        jax_fields.append(JaxField(path=f["path"], type=f["type"], shape=shape_tuple))

    torch_fields: list[TorchField] = []
    for f in torch_fields_json:
        shape_tuple = tuple(
            [int(i) for i in f["shape"].strip("()").split(",") if len(i) > 0]
        )
        torch_fields.append(
            TorchField(path=f["path"], shape=shape_tuple, skip=f["skip"])
        )
    return jax_fields, torch_fields
