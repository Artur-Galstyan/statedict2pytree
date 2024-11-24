import anthropic
import numpy as np
from beartype.typing import Literal, Optional
from loguru import logger

from statedict2pytree.utils.pydantic_models import JaxField, TorchField


PROMPT = """
You will get two lists of strings. These strings are fields of a JAX and PyTorch model.
For example:
--JAX START--
.layers[0].weight
.layers[1].weight
.layers[2].weight
.layers[3].weight
.layers[4].weight
--JAX END--

--PYTORCH START--
layers.0.weight
layers.1.weight
layers.4.weight
layers.2.weight
layers.3.weight
--PYTORCH END--

As you can see, the order doesn't match. This means,
you should look at the PyTorch fields and rearrange them, such that
they match the JAX model. In the above example, the expected return value
is:
--PYTORCH START--
layers.0.weight
layers.1.weight
layers.2.weight
layers.3.weight
layers.4.weight
--PYTORCH END--

Here's another example:
--JAX START--
.conv1.weight
.bn1.weight
.bn1.bias
.bn1.state_index.init[0]
.bn1.state_index.init[1]
--JAX END--

--PYTORCH START--
bn1.running_mean
bn1.running_var
conv1.weight
bn1.weight
bn1.bias
--PYTORCH END--

The expected return value in this case is:

--PYTORCH START--
conv1.weight
bn1.weight
bn1.bias
bn1.running_mean
bn1.running_var
--PYTORCH END--


Sometimes, there are so-called "skip-layers" in the PyTorch model.
Those can be put anywhere, preferably to the end, because your priority
is to match those fields that can be matched first! Here's an example:

--JAX START--
.layers[0].weight
.layers[1].weight
.layers[2].weight
.layers[3].weight
.layers[4].weight
--JAX END--

--PYTORCH START--
layers.0.weight
SKIP
layers.3.weight
layers.2.weight
layers.1.weight
--PYTORCH START--

This should return

--PYTORCH START--
layers.0.weight
layers.1.weight
layers.2.weight
layers.3.weight
SKIP
--PYTORCH START--


It's not always 100% which belongs to which. Use your best judgement.
Start your response with
--PYTORCH START-- and end it with --PYTORCH END--.


Here's your input:
--JAX START--
"""  # pyright: ignore


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


def make_anthropic_request(
    model: Literal["haiku", "opus", "sonnet", "sonnet3.5"],
    content: str,
    api_key: str,
) -> str:
    anthropic_model: Optional[str] = None

    match model:
        case "haiku":
            anthropic_model = "claude-3-haiku-latest"
        case "opus":
            anthropic_model = "claude-3-opus-20240229"
        case "sonnet":
            anthropic_model = "claude-3-sonnet-20240229"
        case "sonnet3.5":
            anthropic_model = "claude-3-5-sonnet-latest"

    logger.info("Creating an instance of the Anthropic client.")
    client = anthropic.Anthropic(api_key=api_key)
    logger.info("Sending a request to the Anthropic API.")
    message = client.messages.create(
        model=anthropic_model,
        max_tokens=4096,
        messages=[{"role": "user", "content": content}],
    )

    return str(message.content[0].text)  # pyright: ignore


def match_using_anthropic(
    jax_fields: list[JaxField],
    torch_fields: list[TorchField],
    model: Literal["haiku", "opus", "sonnet", "sonnet3.5"],
    api_key: str,
) -> Optional[list[TorchField]]:
    prompt = PROMPT
    for jax_field in jax_fields:
        prompt += f"{jax_field.path}\n"
    prompt += "--JAX END--\n"
    prompt += "\n"
    prompt += "--PYTORCH START--\n"
    for torch_field in torch_fields:
        prompt += f"{torch_field.path}\n"
    prompt += "--PYTORCH END--"
    res = make_anthropic_request(model, prompt, api_key)
    lines = res.split("\n")
    rearrangedTorchFields = []
    for i in range(len(lines)):
        matchingTorchField = next(
            (field for field in torch_fields if field.path == lines[i]), None
        )
        if matchingTorchField:
            rearrangedTorchFields.append(matchingTorchField)

    if len(torch_fields) != len(rearrangedTorchFields):
        return None
    return rearrangedTorchFields
