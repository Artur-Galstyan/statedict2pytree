import json
import os
import pathlib
import pickle

import equinox as eqx
import flask
import numpy as np
import torch
from beartype.typing import Literal, Optional
from dotenv import load_dotenv
from jaxtyping import PyTree

from statedict2pytree.converter import (
    convert_from_path,
    convert_from_pytree_and_state_dict,
)
from statedict2pytree.utils.pydantic_models import (
    ChunkifiedPytreePath,
    ChunkifiedStatedictPath,
)
from statedict2pytree.utils.utils import (
    field_jsons_to_fields,
    make_anthropic_request,
)
from statedict2pytree.utils.utils_pytree import pytree_to_fields
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
def match_with_anthropic_endpoint():
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
    anthropic_model: Optional[Literal["haiku", "opus", "sonnet", "sonnet3.5"]] = (
        request_data["model"]
    )
    if not anthropic_model:
        raise ValueError("No model provided")
    res = make_anthropic_request(anthropic_model, content, api_key)

    return json.dumps({"content": res})  # pyright: ignore


def _run_server():
    """
    Start the Flask server on port 5500.

    This function is not meant to be used directly.
    Use the functions `start_conversion_...` functions instead!

    """
    app.run(debug=False, port=5500)
