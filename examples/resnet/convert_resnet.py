import jax
from resnet_model import resnet18
from statedict2pytree.converter import convert
from torchvision.models import resnet18 as t_resnet18, ResNet18_Weights


def convert_resnet():
    resnet_jax = resnet18(key=jax.random.PRNGKey(33), make_with_state=False)
    resnet_torch = t_resnet18(weights=ResNet18_Weights.DEFAULT)
    state_dict = resnet_torch.state_dict()

    # s2p.start_conversion_from_pytree_and_state_dict(resnet_jax, state_dict)

    convert(
        from_memory=True,
        state_dict=state_dict,
        pytree=resnet_jax,
        target_name="resnet18.eqx",
    )


if __name__ == "__main__":
    convert_resnet()
