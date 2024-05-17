import equinox as eqx
import jax
import statedict2pytree as s2p
from resnet import resnet152
from torchvision.models import resnet152 as t_resnet152, ResNet152_Weights


def convert_resnet():
    resnet_jax = resnet152(key=jax.random.PRNGKey(33), make_with_state=False)
    resnet_torch = t_resnet152(weights=ResNet152_Weights.DEFAULT)
    state_dict = resnet_torch.state_dict()

    # s2p.start_conversion(resnet_jax, state_dict)
    model, state = s2p.autoconvert(resnet_jax, state_dict)
    name = "resnet152.eqx"
    eqx.tree_serialise_leaves(name, (model, state))


if __name__ == "__main__":
    convert_resnet()
