import functools as ft
import re

import equinox as eqx
import flask
import jax
import numpy as np
from beartype.typing import Optional
from jaxtyping import PyTree
from loguru import logger
from penzai import pz
from pydantic import BaseModel


app = flask.Flask(__name__)


class Field(BaseModel):
    path: str
    shape: tuple[int, ...]


class TorchField(Field):
    pass


class JaxField(Field):
    type: str


PYTREE: Optional[PyTree] = None
STATE_DICT: Optional[dict] = None


def can_reshape(shape1, shape2):
    product1 = np.prod(shape1)
    product2 = np.prod(shape2)

    return product1 == product2


def get_node(
    tree: PyTree, targets: list[str], log_when_not_found: bool = False
) -> PyTree | None:
    if len(targets) == 0 or tree is None:
        return tree
    else:
        next_target: str = targets[0]
        if bool(re.search(r"\[\d\]", next_target)):
            split_index = next_target.rfind("[")
            name, index = next_target[:split_index], next_target[split_index:]
            index = index[1:-1]
            if hasattr(tree, name):
                subtree = getattr(tree, name)[int(index)]
            else:
                subtree = None
                if log_when_not_found:
                    logger.info(f"Couldn't find  {name} in {tree.__class__}")
        else:
            if hasattr(tree, next_target):
                subtree = getattr(tree, next_target)
            else:
                subtree = None
                if log_when_not_found:
                    logger.info(f"Couldn't find  {next_target} in {tree.__class__}")
        return get_node(subtree, targets[1:])


def pytree_to_fields(pytree: PyTree) -> list[JaxField]:
    flattened, _ = jax.tree_util.tree_flatten_with_path(pytree)
    fields: list[JaxField] = []
    for key_path, value in flattened:
        path = jax.tree_util.keystr(key_path)
        type_path = path.split(".")[1:-1]
        target_path = path.split(".")[1:]
        node_type = type(get_node(pytree, type_path, log_when_not_found=True))
        node = get_node(pytree, target_path, log_when_not_found=True)
        if node is not None and hasattr(node, "shape") and len(node.shape) > 0:
            fields.append(
                JaxField(path=path, type=str(node_type), shape=tuple(node.shape))
            )

    return fields


def state_dict_to_fields(state_dict: Optional[dict]) -> list[TorchField]:
    if state_dict is None:
        return []
    fields: list[TorchField] = []
    for key, value in state_dict.items():
        if hasattr(value, "shape") and len(value.shape) > 0:
            fields.append(TorchField(path=key, shape=tuple(value.shape)))
    return fields


@app.route("/visualize", methods=["POST"])
def visualize_with_penzai():
    global PYTREE, STATE_DICT
    if PYTREE is None or STATE_DICT is None:
        return flask.jsonify({"error": "No Pytree or StateDict found"})
    request_data = flask.request.json
    if request_data is None:
        return flask.jsonify({"error": "No data received"})
    jax_fields = request_data["jaxFields"]
    torch_fields = request_data["torchFields"]
    model, state = convert(jax_fields, torch_fields, PYTREE, STATE_DICT)
    with pz.ts.active_autovisualizer.set_scoped(pz.ts.ArrayAutovisualizer()):
        html_jax = pz.ts.render_to_html((model, state))
        html_torch = pz.ts.render_to_html(STATE_DICT)

    combined_html = f"<html><body>{html_jax}<hr>{html_torch}</body></html>"
    return combined_html


@app.route("/convert", methods=["POST"])
def convert_torch_to_jax():
    global PYTREE, STATE_DICT
    if PYTREE is None or STATE_DICT is None:
        return flask.jsonify({"error": "No Pytree or StateDict found"})
    request_data = flask.request.json
    if request_data is None:
        return flask.jsonify({"error": "No data received"})

    jax_fields_json = request_data["jaxFields"]
    jax_fields: list[JaxField] = []
    for f in jax_fields_json:
        shape_tuple = tuple(
            [int(i) for i in f["shape"].strip("()").split(",") if len(i) > 0]
        )
        jax_fields.append(JaxField(path=f["path"], type=f["type"], shape=shape_tuple))

    torch_fields_json = request_data["torchFields"]
    torch_fields: list[TorchField] = []
    for f in torch_fields_json:
        shape_tuple = tuple(
            [int(i) for i in f["shape"].strip("()").split(",") if len(i) > 0]
        )
        torch_fields.append(TorchField(path=f["path"], shape=shape_tuple))

    name = request_data["name"]
    model, state = convert(jax_fields, torch_fields, PYTREE, STATE_DICT)
    eqx.tree_serialise_leaves(name, (model, state))

    return flask.jsonify({"status": "success"})


@app.route("/", methods=["GET"])
def index():
    pytree_fields = pytree_to_fields(PYTREE)
    return flask.render_template(
        "index.html",
        pytree_fields=pytree_fields,
        torch_fields=state_dict_to_fields(STATE_DICT),
    )


def autoconvert(pytree: PyTree, state_dict: dict) -> tuple[PyTree, eqx.nn.State]:
    jax_fields = pytree_to_fields(pytree)
    torch_fields = state_dict_to_fields(state_dict)
    return convert(jax_fields, torch_fields, pytree, state_dict)


def convert(
    jax_fields: list[JaxField],
    torch_fields: list[TorchField],
    pytree: PyTree,
    state_dict: dict,
) -> tuple[PyTree, eqx.nn.State]:
    identity = lambda *args, **kwargs: pytree
    model, state = eqx.nn.make_with_state(identity)()
    state_paths: list[tuple[JaxField, TorchField]] = []
    for jax_field, torch_field in zip(jax_fields, torch_fields):
        if not can_reshape(jax_field.shape, torch_field.shape):
            raise ValueError(
                "Fields have incompatible shapes!"
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


def start_conversion(pytree: PyTree, state_dict: dict):
    global PYTREE, STATE_DICT
    if state_dict is None:
        raise ValueError("STATE_DICT must not be None!")
    PYTREE = pytree
    STATE_DICT = state_dict

    for k, v in STATE_DICT.items():
        STATE_DICT[k] = v.numpy()
    app.run(debug=True, port=5500)
