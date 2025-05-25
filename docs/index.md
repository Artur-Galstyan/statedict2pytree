# Quickstart Guide

## Installation

To install `statedict2pytree`, run:

```bash
pip install statedict2pytree
```

---

## Basic Usage

There are 4-5 main functions you might interact with:

* `autoconvert`
* `convert`
* `pytree_to_fields`
* `state_dict_to_fields`
* `move_running_fields_to_the_end` (optional helper)

---

## General Information

`statedict2pytree` primarily aligns your JAX PyTree and the PyTorch `state_dict` side-by-side. It then checks if the shapes of the aligned weights match. If they do, it converts the PyTorch tensors to JAX arrays and places them into a new PyTree with the same structure as your original JAX PyTree.

**This means that the order and the shape of the arrays in your PyTree and the `state_dict` must match after any optional reordering!** The `pytree_to_fields` function uses a filter (defaulting to `equinox.is_array`) to determine which elements are considered fields.

For example, this conversion will **work** ✅:

| Parameter         | JAX Shape    | PyTorch Shape |
| :---------------- | :----------- | :------------ |
| `linear.weight`   | `(2, 2)`     | `(2, 2)`      |
| `linear.bias`     | `(2,)`       | `(2,)`        |
| `conv.weight`     | `(1, 1, 2, 2)` | `(1, 1, 2, 2)`|
| `conv.bias`       | `(1,)`       | `(1,)`        |

Since the shapes match when aligned in the same order, the conversion is successful.

On the other hand, this will **not work** ❌:

| Parameter         | JAX Shape    | PyTorch Shape | Mismatch? |
| :---------------- | :----------- | :------------ | :-------- |
| `linear.weight`   | `(2, 2)`     | `(3, 2)`      | Yes       |
| `linear.bias`     | `(2,)`       | `(3,)`        | Yes       |
| `conv.weight`     | `(1, 1, 2, 2)` | `(1, 1, 2, 2)`| No        |
| `conv.bias`       | `(1,)`       | `(1,)`        | No        |

This conversion will fail because the shapes of `model.linear.weight` and `model.linear.bias` don't match between the PyTree and the state dict.

Another reason why the conversion might fail is if the **order** of parameters (and thus the shapes of misaligned parameters) doesn't match:

| JAX Parameter (Model Order) | JAX Shape      | PyTorch Counterpart (`state_dict` Order) | PyTorch Shape  | Issue if Matched Sequentially                     |
| :-------------------------- | :------------- | :--------------------------------------- | :------------- | :------------------------------------------------ |
| `model['conv']['weight']`   | `(1, 1, 2, 2)` | `state_dict['model.linear.weight']`      | `(2, 2)`       | Order: JAX `conv.w` `(1122)` vs PT `linear.w` `(22)` |
| `model['conv']['bias']`     | `(1,)`         | `state_dict['model.linear.bias']`        | `(2,)`         | Order: JAX `conv.b` `(1,)` vs PT `linear.b` `(2,)`   |
| `model['linear']['weight']` | `(2, 2)`       | `state_dict['model.conv.weight']`        | `(1, 1, 2, 2)` | Order: JAX `linear.w` `(22)` vs PT `conv.w` `(1122)`|
| `model['linear']['bias']`   | `(2,)`         | `state_dict['model.conv.bias']`          | `(1,)`         | Order: JAX `linear.b` `(2,)` vs PT `conv.b` `(1,)`   |

To help with the order issue, you can provide a `list[str]` specifying the desired order of PyTree fields (matching the `state_dict`'s conceptual order, or vice-versa if you reorder `state_dict` fields). This is especially helpful when you can't easily force the correct order using `move_running_fields_to_the_end`. For the example above, if your PyTree expects `conv` then `linear`, the list of strings representing the *names from the state\_dict in the JAX PyTree's desired order* would be:

```python
['model.conv.weight', 'model.conv.bias', 'model.linear.weight', 'model.linear.bias']
```
This list would be passed to `pytree_to_fields` via `autoconvert`'s `pytree_model_order` argument to ensure `jaxfields` are in this sequence. Alternatively, you could reorder `torchfields` using `move_running_fields_to_the_end` or other custom logic.

---

## API Reference

### `autoconvert`

This is the simplest, highest-level function for most use cases.

```python
def autoconvert(
    pytree: PyTree,
    state_dict: dict,
    pytree_model_order: list[str] | None = None
) -> PyTree:
    ...
```

You provide your JAX `pytree` and the PyTorch `state_dict`. Optionally, you can give `pytree_model_order` (a list of strings representing `jax.tree_util.keystr(path)`) to ensure the JAX fields are processed in a specific sequence. It handles the steps of field extraction (using `pytree_to_fields` with its default `filter=eqx.is_array`), alignment, and conversion, returning the populated JAX PyTree. If you need custom filtering for PyTree leaves, you should use `pytree_to_fields` and `convert` separately.

* **Parameters**:
    * `pytree`: The JAX PyTree (e.g., an Equinox model) whose structure is the target.
    * `state_dict`: The PyTorch state dictionary containing the weights.
    * `pytree_model_order` (optional): A list of JAX KeyPath strings (like `'.layers.0.linear.weight'`). If provided, JAX fields will be ordered according to this list. This is useful if the automatic PyTree traversal order doesn't match the `state_dict` order.
* **Returns**: A new JAX PyTree with the same structure as the input `pytree`, but with weights populated from the `state_dict`.

---

### `convert`

This is the core function that performs the actual conversion once the JAX PyTree fields and PyTorch `state_dict` fields have been extracted and aligned.

```python
def convert(
    state_dict: dict[str, Any],
    pytree: PyTree,
    jaxfields: list[JaxField],
    state_indices: dict | None,
    torchfields: list[TorchField],
    dtype: Any | None = None,
) -> PyTree:
    ...
```

It iterates through the aligned `jaxfields` and `torchfields`, checks for shape compatibility (reshapability), converts PyTorch tensors (expected as values in `state_dict`) to JAX arrays (optionally casting `dtype`), and inserts them into the correct place in the JAX PyTree.

* **Parameters**:
    * `state_dict`: The original PyTorch state dictionary. Values are expected to be tensor-like (e.g., `torch.Tensor`).
    * `pytree`: The JAX PyTree that will be populated.
    * `jaxfields`: An ordered list of `JaxField` objects (obtained from `pytree_to_fields`) representing the leaves of the JAX PyTree.
    * `state_indices`: A dictionary mapping state markers to `eqx.nn.StateIndex` objects, used for handling Equinox stateful layers.
    * `torchfields`: An ordered list of `TorchField` objects (obtained from `state_dict_to_fields`) representing the tensors in the PyTorch `state_dict`. **This list must be ordered to match `jaxfields`**.
    * `dtype` (optional): The JAX data type to convert floating-point tensors to (e.g., `jnp.float32`). Defaults to JAX's current default floating-point type.
* **Returns**: A new JAX PyTree populated with weights from the `state_dict`.

---

### `pytree_to_fields`

This function traverses a JAX PyTree and extracts information about its array leaves based on a filter.

```python
def pytree_to_fields(
    pytree: PyTree,
    model_order: list[str] | None = None,
    filter: Callable[[Array], bool] = eqx.is_array,
) -> tuple[list[JaxField], dict | None]:
    ...
```

It identifies all JAX arrays (or other elements satisfying the `filter`) within the `pytree`, recording their `KeyPath` (path within the PyTree) and shape. If `model_order` is provided, it attempts to reorder the extracted fields according to that list. This is crucial for ensuring the JAX fields align correctly with the PyTorch fields.

* **Parameters**:
    * `pytree`: The JAX PyTree to analyze.
    * `model_order` (optional): A list of strings, where each string is a `jax.tree_util.keystr` representation of a `KeyPath` to an array leaf in the `pytree`. If provided, the output `JaxField` list will be sorted according to this order, with any fields not in `model_order` appended at the end.
    * `filter` (optional): A callable that takes a PyTree leaf (e.g., an array) and returns `True` if it should be considered a field to be converted, `False` otherwise. Defaults to `equinox.is_array`.
* **Returns**: A tuple containing:
    * `list[JaxField]`: A list of `JaxField` objects, each describing a filtered leaf in the PyTree (path, shape).
    * `dict | None`: A dictionary containing information about `eqx.nn.StateIndex` objects found in the PyTree, or `None` if none are found.

---

### `state_dict_to_fields`

This function processes a PyTorch `state_dict` to extract information about its tensors.

```python
def state_dict_to_fields(
    state_dict: dict[str, Any],
) -> list[TorchField]:
    ...
```

It iterates through the `state_dict`, creating a `TorchField` object for each value that has a `shape` attribute and a non-empty shape (typically tensors). This object stores the tensor's name (key in the `state_dict`) and its shape.

* **Parameters**:
    * `state_dict`: The PyTorch state dictionary. Values are typically `torch.Tensor` or other array-like objects.
* **Returns**: A list of `TorchField` objects, each describing a tensor in the `state_dict` (path/key, shape). The order matches the iteration order of the input `state_dict`.

---

### `move_running_fields_to_the_end`

This is an optional utility function to help reorder fields extracted from a PyTorch `state_dict`.

```python
def move_running_fields_to_the_end(
    torchfields: list[TorchField],
    identifier: str = "running_"
):
    ...
```

It's particularly useful for models with layers like `BatchNorm`, where PyTorch often stores `running_mean` and `running_var` interspersed with weights and biases, while Equinox (a common JAX library) typically expects stateful components like these at the end of a layer's parameter list. This function moves any `TorchField` whose path contains the `identifier` (defaulting to `"running_"`) to the end of the list.

* **Parameters**:
    * `torchfields`: The list of `TorchField` objects to be reordered.
    * `identifier` (optional): A string that, if found within a `TorchField`'s path, will cause that field to be moved to the end of the list. Default is `"running_"`.
* **Returns**: The modified list of `TorchField` objects with identified fields moved to the end.
