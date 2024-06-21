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

from statedict2pytree.utils.pydantic_models import JaxField, TorchField
from statedict2pytree.utils.utils import can_reshape, field_jsons_to_fields
from statedict2pytree.utils.utils_pytree import get_node, pytree_to_fields
from statedict2pytree.utils.utils_state_dict import state_dict_to_fields


load_dotenv()

app = flask.Flask(__name__, static_url_path="/", static_folder="../client/public")


PYTREE: Optional[PyTree] = None
PYTREE_PATH: Optional[str] = None

STATE_DICT: Optional[dict[str, np.ndarray]] = None
STATE_DICT_PATH: Optional[str] = None


@app.route("/", methods=["GET"])
def index():
    return flask.send_from_directory("../client/public", "index.html")


@app.route("/startup/getTorchFields", methods=["GET"])
def _get_torch_fields():
    if STATE_DICT is None:
        if STATE_DICT_PATH is None:
            raise ValueError("STATE_DICT None AND STATE_DICT_PATH was not provided!")
        with open(str(pathlib.Path(STATE_DICT_PATH) / "torch_fields.pkl"), "rb") as f:
            torch_fields = pickle.load(f)
    else:
        torch_fields = state_dict_to_fields(STATE_DICT)
    fields = []
    for tf in torch_fields:
        fields.append(tf.model_dump())
    return fields


@app.route("/startup/getJaxFields", methods=["GET"])
def _get_jax_fields():
    if PYTREE is None:
        if PYTREE_PATH is None:
            raise ValueError("PYTREE is None AND PYTREE_PATH was not provided!")
        with open(str(pathlib.Path(PYTREE_PATH) / "jax_fields.pkl"), "rb") as f:
            jax_fields = pickle.load(f)
    else:
        jax_fields = pytree_to_fields(PYTREE)
    fields = []
    for jf in jax_fields:
        fields.append(jf.model_dump())
    return fields


@app.route("/convert", methods=["POST"])
def _convert_torch_to_jax():
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
    jax_fields = pytree_to_fields(pytree)
    torch_fields = state_dict_to_fields(state_dict)

    for k, v in state_dict.items():
        state_dict[k] = v.numpy()
    return convert_from_pytree_and_state_dict(
        jax_fields, torch_fields, pytree, state_dict
    )


def autoconvert_from_paths(
    pytree_path: str,
    state_dict_path: str,
):
    with open(str(pathlib.Path(pytree_path) / "jax_fields.pkl"), "rb") as f:
        jax_fields = pickle.load(f)

    with open(str(pathlib.Path(state_dict_path) / "torch_fields.pkl"), "rb") as f:
        torch_fields = pickle.load(f)
    convert_from_path(jax_fields, torch_fields, pytree_path, state_dict_path)


def convert_from_path(
    jax_fields: list[JaxField],
    torch_fields: list[TorchField],
    pytree_path: str,
    state_dict_path: str,
):
    j_path = pathlib.Path(pytree_path)
    t_path = pathlib.Path(state_dict_path)

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
    global PYTREE_PATH, STATE_DICT_PATH
    PYTREE_PATH = pytree_path
    STATE_DICT_PATH = state_dict_path

    run_server()


def start_conversion_from_pytree_and_state_dict(
    pytree: PyTree, state_dict: dict[str, torch.Tensor]
):
    global PYTREE, STATE_DICT
    PYTREE = pytree
    STATE_DICT = dict()

    for k, v in state_dict.items():
        STATE_DICT[k] = v.numpy()
    run_server()


@app.post("/anthropic")
def make_anthropic_request():
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
    if not anthropic_model:
        return flask.jsonify({"error": "No model provided"})

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=anthropic_model,
        max_tokens=4096,
        messages=[{"role": "user", "content": content}],
    )

    return json.dumps({"content": str(message.content[0].text)})  # pyright: ignore


def run_server():
    app.run(debug=False, port=5500)
