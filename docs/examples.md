# Usage Examples

## Converting LLaMA 3 Model

This example demonstrates how to convert the LLaMA 3 model from PyTorch to JAX:
(see the `examples` directory on GitHub)

```python
import json
import os
import tempfile
import jax
import torch
from jaxonmodels.transformers.llama.llama3 import LLaMA
from jaxonmodels.transformers.llama.model_args import LLaMAModelArgs
from loguru import logger
from statedict2pytree.statedict2pytree import start_conversion_from_paths
from statedict2pytree.utils.utils_pytree import chunkify_pytree, serialize_pytree_chunks
from statedict2pytree.utils.utils_state_dict import chunkify_state_dict
from tqdm import tqdm

def main():
    with tempfile.TemporaryDirectory() as tmp:
        logger.info("Loading state dict...")
        state_dict = torch.load("Meta-Llama-3-8B/consolidated.00.pth")
        logger.info("Done. Converting to float16...")
        for key in tqdm(state_dict.keys()):
            if torch.is_tensor(state_dict[key]):
                state_dict[key] = state_dict[key].to(torch.float16)
        logger.info("Done. Chunkifying state dict...")
        chunkify_state_dict(state_dict, tmp)
        logger.info("Done. Deleting state dict...")
        del state_dict
        with open("Meta-Llama-3-8B/params.json", "r") as f:
            params = json.load(f)
        model_args = LLaMAModelArgs(**params)
        model_args.precision = "quarter"
        key = jax.random.PRNGKey(21)
        logger.info("Creating JAX model...")
        model = LLaMA(model_args, key=key)
        logger.info("Done. Chunkifying PyTree...")
        paths = chunkify_pytree(model, tmp)
        logger.info("Done. Starting server...")
        start_conversion_from_paths(tmp, tmp)
        paths = os.listdir(f"{tmp}/pytree")
        tree_paths = []
        for p in paths:
            tree_paths.append(f"{tmp}/pytree/" + p)
        logger.info("SERIALIZING PYTREE")
        serialize_pytree_chunks(model, tree_paths, "model.eqx")
        logger.info("Done.")

if __name__ == "__main__":
    main()
```

## Converting ResNet18 Model
This example shows how to convert a ResNet18 model from PyTorch to JAX:

(see the `examples` directory on GitHub)

```python
import jax
import statedict2pytree as s2p
from resnet_model import resnet18
from torchvision.models import resnet18 as t_resnet18, ResNet18_Weights

def convert_resnet():
    resnet_jax = resnet18(key=jax.random.PRNGKey(33), make_with_state=False)
    resnet_torch = t_resnet18(weights=ResNet18_Weights.DEFAULT)
    state_dict = resnet_torch.state_dict()
    s2p.start_conversion_from_pytree_and_state_dict(resnet_jax, state_dict)

if __name__ == "__main__":
    convert_resnet()
```
