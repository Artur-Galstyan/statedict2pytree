import os
import sys

import jax
from torchvision.models import resnet18 as t_resnet18, ResNet18_Weights


# add examples folder to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "examples"))

# import it
from examples.resnet.resnet_model import resnet18
from statedict2pytree.converter import convert


resnet_jax = resnet18(key=jax.random.PRNGKey(33), make_with_state=False)
resnet_torch = t_resnet18(weights=ResNet18_Weights.DEFAULT)
state_dict = resnet_torch.state_dict()
if __name__ == "__main__":
    print("Converting from PyTree and state_dict...")
    convert(
        from_memory=True,
        path_to_pytree_object="test_example.resnet_jax",
        path_to_state_dict_object="test_example.state_dict",
        arrange_with_anthropic=True,
    )
