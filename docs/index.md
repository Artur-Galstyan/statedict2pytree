# Quickstart Guide

## Installation

To install StateDict2PyTree, run:

```bash
pip install statedict2pytree
```

## Basic Usage
StateDict2PyTree provides two ways to convert PyTorch state dicts to JAX pytrees, depending on if the model fits in your memory or not.

The main function is the `convert` function, which can be imported from `from statedict2pytree.converter import convert` and has these arguments:

```python
def convert(
    from_memory: Optional[bool] = False,
    from_path: Optional[bool] = False,
    state_dict: Optional[dict[str, torch.Tensor]] = None,
    pytree: Optional[PyTree] = None,
    path_to_state_dict_object: Optional[str] = None,
    path_to_pytree_object: Optional[str] = None,
    chunkify: bool = False,
    path_to_state_dict_chunks: Optional[str] = None,
    path_to_pytree_chunks: Optional[str] = None,
    arrange_with_anthropic: bool = False,
    anthropic_model: Literal["haiku", "opus", "sonnet", "sonnet3.5"] = "sonnet3.5",
    target_name: str = "model.eqx",
    target_dir: str = ".",
):
    ...
```

Depending on which mode you're using (`from_memory` or `from_path`), different
arguments become required. To transform from memory, you need to provide these
arguments:

```
    state_dict: dict[str, torch.Tensor],
    pytree: PyTree,
    arrange_with_anthropic: bool,
    anthropic_model: str,
    target_dir: str,
    target_name: str,
```

(Note that the default parameters are applied)

If your model does not fit on memory, you would use `from_path=True`. Here,
the program splits into 2 paths: either you chunkify the model yourself and
store them somewhere and provide the path, or you let the program chunkify
your models in a `tempdir` (meaning the chunks are deleted afterwards).

You will need these arguments in that case:
```
path_to_state_dict_chunks: Optional[str],
path_to_state_dict_object: Optional[str],
chunkify: bool,
path_to_pytree_chunks: Optional[str],
path_to_pytree_object: Optional[str],
arrange_with_anthropic: bool,
anthropic_model: str,
target_dir: str,
target_name: str,
```

What's important here is to know if your models weights and the weights in the
state dict are in the *right order*! This might not always be the case.

For example, in this case, the conversion will fail:


PyTree:
lin1 - Shape: 10, 20
lin2 - Shape: 20, 20

State Dict:
lin2 - Shape: 20, 20
lin1 - Shape: 10, 20

This will fail, because `s2p` will look at the order of the weights and in this case, the shapes don't match.

You might be wondering now: "why not just match the shapes?" Unfortunately,
it's not guaranteed that the shapes are unique. Consider this case:


PyTree:
lin1 - Shape: 10, 10
lin2 - Shape: 10, 10
lin3 - Shape: 10, 10


State Dict:
lin3 - Shape: 10, 10
lin2 - Shape: 10, 10
lin1 - Shape: 10, 10

Conversion in this case would succeed, but the resulting model would be incorrect. We could try to infer the order with the names, but after trying
a couple of algorithms to match the names, nothing really worked. This is why
you can either:

- use the provided GUI to align the weights yourself
- or use Anthropic's Claude model to align them, without starting the GUI

To start the GUI, you will need to do the following:


```python
from statedict2pytree.app import start_conversion_from_pytree_and_state_dict
from statedict2pytree.app import start_conversion_from_paths

# if in memory:
start_conversion_from_pytree_and_state_dict(
    pytree: PyTree, state_dict: dict[str, torch.Tensor]
)

# else:
start_conversion_from_paths(pytree_path: str, state_dict_path: str)
```

This will start a Flask server and you can start aligning your model.

On the other hand, you can also provide your `ANTHROPIC_API_KEY` in your
environment variables and set `arrange_with_anthropic` to `True` in the main `convert` function. It will ask you to confirm if you like the arrangement it made (usually it works on the first try).
