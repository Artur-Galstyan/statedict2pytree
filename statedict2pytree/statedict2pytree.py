import functools as ft
import json
import os
import pathlib
import pickle

import anthropic
import equinox as eqx
import flask
import numpy as np
import torch
from beartype.typing import Literal, Optional
from dotenv import load_dotenv
from jaxtyping import PyTree
from tqdm import tqdm

from statedict2pytree.utils.pydantic_models import (
    ChunkifiedPytreePath,
    ChunkifiedStatedictPath,
    JaxField,
    TorchField,
)
from statedict2pytree.utils.utils import can_reshape, field_jsons_to_fields
from statedict2pytree.utils.utils_pytree import get_node, pytree_to_fields
from statedict2pytree.utils.utils_state_dict import state_dict_to_fields


load_dotenv()

app = flask.Flask(__name__, static_url_path="/", static_folder="../client/public")


PYTREE: Optional[PyTree] = None
PYTREE_PATH: Optional[ChunkifiedPytreePath] = None

STATE_DICT: Optional[dict[str, np.ndarray]] = None
STATE_DICT_PATH: Optional[ChunkifiedStatedictPath] = None


@app.route("/", methods=["GET"])
def index():
    """
    Serve the index.html file from the client's public directory.

    Returns:
        The contents of index.html file.
    """
    return flask.send_from_directory("../client/public", "index.html")


@app.route("/startup/getTorchFields", methods=["GET"])
def _get_torch_fields():
    """
    Retrieve the TorchFields from either the loaded STATE_DICT or from a pickle file.

    This function is not meant to be imported/used directly!

    Returns:
        list: A list of dictionaries representing TorchField objects.

    Raises:
        ValueError: If both STATE_DICT and STATE_DICT_PATH are None.
    """
    if STATE_DICT is None:
        if STATE_DICT_PATH is None:
            raise ValueError("STATE_DICT None AND STATE_DICT_PATH was not provided!")
        with open(
            str(pathlib.Path(STATE_DICT_PATH.path) / "torch_fields.pkl"), "rb"
        ) as f:
            torch_fields = pickle.load(f)
    else:
        torch_fields = state_dict_to_fields(STATE_DICT)
    fields = []
    for tf in torch_fields:
        fields.append(tf.model_dump())
    return fields


@app.route("/startup/getJaxFields", methods=["GET"])
def _get_jax_fields():
    """
    Retrieve the JaxFields from either the loaded PYTREE or from a pickle file.

    This function is not meant to be imported/used directly!

    Returns:
        list: A list of dictionaries representing JaxField objects.

    Raises:
        ValueError: If both PYTREE and PYTREE_PATH are None.
    """
    if PYTREE is None:
        if PYTREE_PATH is None:
            raise ValueError("PYTREE is None AND PYTREE_PATH was not provided!")
        with open(str(pathlib.Path(PYTREE_PATH.path) / "jax_fields.pkl"), "rb") as f:
            jax_fields = pickle.load(f)
    else:
        jax_fields = pytree_to_fields(PYTREE)
    fields = []
    for jf in jax_fields:
        fields.append(jf.model_dump())
    return fields


@app.route("/convert", methods=["POST"])
def _convert_torch_to_jax():
    """
    Convert PyTorch state dict to JAX pytree based on provided field mappings.

    This function handles conversion either from memory or from file paths,
    depending on the current state of global variables.

    This function is not meant to be imported/used directly!

    Returns:
        flask.Response: JSON response indicating success or error.

    Raises:
        ValueError: If required global variables are not properly set.
    """
    global PYTREE, STATE_DICT
    global PYTREE_PATH, STATE_DICT_PATH
    mode: Optional[Literal["FROM_PATH", "FROM_MEMORY"]] = None
    if (
        PYTREE is None
        and STATE_DICT is None
        and PYTREE_PATH is not None
        and STATE_DICT_PATH is not None
    ):
        mode = "FROM_PATH"
    elif (
        PYTREE is not None
        and STATE_DICT is not None
        and STATE_DICT_PATH is None
        and PYTREE_PATH is None
    ):
        mode = "FROM_MEMORY"
    else:
        return flask.jsonify({"error": "No Pytree or StateDict found"})
    request_data = flask.request.json
    if request_data is None:
        return flask.jsonify({"error": "No data received"})

    jax_fields, torch_fields = field_jsons_to_fields(
        request_data["jaxFields"], request_data["torchFields"]
    )

    name = request_data["model"]

    if mode == "FROM_MEMORY":
        if STATE_DICT is None or PYTREE is None:
            raise ValueError("STATE_DICT or PYTREE is None")
        model, state = convert_from_pytree_and_state_dict(
            jax_fields, torch_fields, PYTREE, STATE_DICT
        )
        eqx.tree_serialise_leaves(name, (model, state))
    else:
        if STATE_DICT_PATH is None or PYTREE_PATH is None:
            raise ValueError("STATE_DICT_PATH or PYTREE_PATH is None")
        convert_from_path(jax_fields, torch_fields, PYTREE_PATH, STATE_DICT_PATH)
    return flask.jsonify({"status": "success"})


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


def start_conversion_from_paths(pytree_path: str, state_dict_path: str):
    """
    Initialize the conversion process using file paths and start the Flask server.

    Args:
        pytree_path (str): Path to the JAX pytree directory.
        state_dict_path (str): Path to the PyTorch state dict directory.
    """
    global PYTREE_PATH, STATE_DICT_PATH
    PYTREE_PATH = ChunkifiedPytreePath(path=pytree_path)
    STATE_DICT_PATH = ChunkifiedStatedictPath(path=state_dict_path)
    _run_server()


def start_conversion_from_pytree_and_state_dict(
    pytree: PyTree, state_dict: dict[str, torch.Tensor]
):
    """
    Initialize the conversion process using in-memory structures and
    start the Flask server.

    Args:
        pytree (PyTree): The target JAX pytree structure.
        state_dict (dict[str, torch.Tensor]): The source PyTorch state dict.
    """
    global PYTREE, STATE_DICT
    PYTREE = pytree
    STATE_DICT = dict()

    for k, v in state_dict.items():
        STATE_DICT[k] = v.numpy()
    _run_server()


@app.post("/anthropic")
def make_anthropic_request():
    """
    Make a request to the Anthropic API using the provided content and model.

    Returns:
        flask.Response: JSON response containing the API response or an error message.

    Raises:
        KeyError: If required keys are missing in the request data.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", None)
    if api_key is None:
        return {"error": "ANTHROPIC_API_KEY not set in env vars!"}

    request_data = flask.request.json
    if request_data is None:
        return flask.jsonify({"error": "No data received"})
    if "content" not in request_data:
        return flask.jsonify({"error": "No data received"})

    if "model" not in request_data:
        return flask.jsonify({"error": "There was no model provided"})

    content = request_data["content"]

    anthropic_model: Optional[str] = None

    match request_data["model"]:
        case "haiku":
            anthropic_model = "claude-3-haiku-20240307"
        case "opus":
            anthropic_model = "claude-3-opus-20240229"
        case "sonnet":
            anthropic_model = "claude-3-sonnet-20240229"
        case "sonnet3.5":
            anthropic_model = "claude-3-5-sonnet-20240620"
    if not anthropic_model:
        return flask.jsonify({"error": "No model provided"})

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=anthropic_model,
        max_tokens=4096,
        messages=[{"role": "user", "content": content}],
    )

    return json.dumps({"content": str(message.content[0].text)})  # pyright: ignore


def _run_server():
    """
    Start the Flask server on port 5500.

    This function is not meant to be used directly.
    Use the functions `start_conversion_...` functions instead!

    """
    app.run(debug=False, port=5500)
