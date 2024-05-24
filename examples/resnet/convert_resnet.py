import jax
from resnet import resnet18
from torchvision.models import resnet18 as t_resnet18, ResNet18_Weights


def convert_resnet():
    resnet_jax = resnet18(key=jax.random.PRNGKey(33), make_with_state=False)
    resnet_torch = t_resnet18(weights=ResNet18_Weights.DEFAULT)
    state_dict = resnet_torch.state_dict()

    # model, state = s2p.autoconvert(resnet_jax, state_dict)
    # name = "resnet18.eqx"
    # eqx.tree_serialise_leaves(name, (model, state))


if __name__ == "__main__":
    convert_resnet()
