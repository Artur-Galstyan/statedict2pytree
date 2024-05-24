import os
import tempfile

import jax
from examples.resnet.resnet import resnet50
from statedict2pytree.statedict2pytree import (
    autoconvert_from_paths,
)
from statedict2pytree.utils.utils_pytree import chunkify_pytree, serialize_pytree_chunks
from statedict2pytree.utils.utils_state_dict import chunkify_state_dict
from torchvision.models import resnet50 as t_resnet50, ResNet50_Weights


with tempfile.TemporaryDirectory() as tmp:
    resnet_torch = t_resnet50(weights=ResNet50_Weights.DEFAULT)
    state_dict = resnet_torch.state_dict()
    chunkify_state_dict(state_dict, tmp)
    resnet_jax = resnet50(key=jax.random.PRNGKey(33), make_with_state=False)
    paths = chunkify_pytree(resnet_jax, tmp)
    # start_conversion_from_paths(tmp, tmp)
    autoconvert_from_paths(pytree_path=tmp, state_dict_path=tmp)
    paths = os.listdir(f"{tmp}/pytree")
    tree_paths = []
    for p in paths:
        tree_paths.append(f"{tmp}/pytree/" + p)
    print("SERIALIZING PYTREE")
    serialize_pytree_chunks(resnet_jax, tree_paths, "model.eqx")
