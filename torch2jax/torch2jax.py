import re

import equinox as eqx
import fire
import functools as ft
import flask
import jax
from beartype.typing import Optional
from jaxtyping import PyTree
from loguru import logger
from pydantic import BaseModel


app = flask.Flask(__name__)


class Field(BaseModel):
    path: str
    type: str
    shape: tuple[int, ...]


PYTREE: Optional[PyTree] = None
STATE_DICT: Optional[dict] = None


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


def pytree_to_fields(pytree: PyTree) -> list[Field]:
    flattened, _ = jax.tree_util.tree_flatten_with_path(pytree)
    fields: list[Field] = []
    for key_path, value in flattened:
        path = jax.tree_util.keystr(key_path)
        type_path = path.split(".")[1:-1]
        target_path = path.split(".")[1:]
        node_type = type(get_node(pytree, type_path, log_when_not_found=True))
        node = get_node(pytree, target_path, log_when_not_found=True)
        if node is not None and hasattr(node, "shape") and len(node.shape) > 0:
            fields.append(
                Field(path=path, type=str(node_type), shape=tuple(node.shape))
            )

    return fields


def state_dict_to_fields(state_dict: Optional[dict]) -> list[Field]:
    if state_dict is None:
        return []
    fields: list[Field] = []
    for key, value in state_dict.items():
        if hasattr(value, "shape") and len(value.shape) > 0:
            fields.append(
                Field(path=key, type=str(type(value)), shape=tuple(value.shape))
            )
    return fields


@app.route("/convert", methods=["POST"])
def convert_torch_to_jax():
    global PYTREE, STATE_DICT
    if PYTREE is None or STATE_DICT is None:
        return flask.jsonify({"error": "No Pytree or StateDict found"})
    request_data = flask.request.json
    if request_data is None:
        return flask.jsonify({"error": "No data received"})
    jax_fields = request_data["jaxFields"]
    torch_fields = request_data["torchFields"]

    identity = lambda *args, **kwargs: PYTREE
    model, state = eqx.nn.make_with_state(identity)()
    state_paths = []
    for jax_field, torch_field in zip(jax_fields, torch_fields):
        path = jax_field["path"].split(".")[1:]
        if "StateIndex" in jax_field["type"]:
            state_paths.append((jax_field, torch_field))

        else:
            where = ft.partial(get_node, targets=path)
            if where(model) is not None:
                logger.info(f"Found {jax_field['path']} in PyTree, updating...")
                model = eqx.tree_at(
                    where,
                    model,
                    STATE_DICT[torch_field["path"]].numpy(),
                )
    result = {}
    for tuple_item in state_paths:
        path_prefix = tuple_item[0]["path"].split(".")[1:-1]
        prefix_key = ".".join(path_prefix)
        if prefix_key not in result:
            result[prefix_key] = []
        result[prefix_key].append(tuple_item[1])

    for key in result:
        state_index = get_node(model, key.split("."))
        if state_index is not None:
            to_replace_tuple = tuple(
                [STATE_DICT[i["path"]].numpy() for i in result[key]]
            )
            print(to_replace_tuple)
            state = state.set(state_index, to_replace_tuple)

    # serialize the result

    return flask.jsonify({"error": "Not implemented yet"})


@app.route("/", methods=["GET"])
def main():
    pytree_fields = pytree_to_fields(PYTREE)
    return flask.render_template(
        "index.html",
        pytree_fields=pytree_fields,
        torch_fields=state_dict_to_fields(STATE_DICT),
    )


def convert(pytree: PyTree, state_dict: dict):
    global PYTREE, STATE_DICT
    PYTREE = pytree
    STATE_DICT = state_dict
    app.run(debug=True, port=5500)


def start_server(debug: bool = True, port: int = 5500):
    app.run(debug=debug, port=port)


if __name__ == "__main__":
    fire.Fire(start_server)
