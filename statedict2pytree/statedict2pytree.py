import functools as ft
import os
import pathlib
import pickle

import equinox as eqx
import flask
import numpy as np
import torch
from beartype.typing import Literal, Optional
from jaxtyping import PyTree

from statedict2pytree.utils.pydantic_models import JaxField, TorchField
from statedict2pytree.utils.utils import can_reshape, field_jsons_to_fields
from statedict2pytree.utils.utils_pytree import get_node, pytree_to_fields
from statedict2pytree.utils.utils_state_dict import state_dict_to_fields


app = flask.Flask(__name__)


PYTREE: Optional[PyTree] = None
PYTREE_PATH: Optional[str] = None

STATE_DICT: Optional[dict[str, np.ndarray]] = None
STATE_DICT_PATH: Optional[str] = None


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

    name = request_data["name"]

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


@app.route("/", methods=["GET"])
def index():
    if PYTREE is None:
        if PYTREE_PATH is None:
            raise ValueError("PYTREE is None AND PYTREE_PATH was not provided!")
        with open(str(pathlib.Path(PYTREE_PATH) / "jax_fields.pkl"), "rb") as f:
            jax_fields = pickle.load(f)
    else:
        jax_fields = pytree_to_fields(PYTREE)

    if STATE_DICT is None:
        if STATE_DICT_PATH is None:
            raise ValueError("STATE_DICT None AND STATE_DICT_PATH was not provided!")
        with open(str(pathlib.Path(STATE_DICT_PATH) / "torch_fields.pkl"), "rb") as f:
            torch_fields = pickle.load(f)
    else:
        torch_fields = state_dict_to_fields(STATE_DICT)

    return flask.render_template(
        "index.html", pytree_fields=jax_fields, torch_fields=torch_fields
    )


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

    for jax_field, torch_field in zip(jax_fields, torch_fields):
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


def run_server():
    app.jinja_env.globals.update(enumerate=enumerate)
    app.run(debug=False, port=5500)
