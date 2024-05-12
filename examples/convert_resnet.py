import jax
import statedict2pytree as s2p
from resnet import resnet50
from torchvision.models import resnet50 as t_resnet50, ResNet50_Weights


def convert_resnet():
    resnet_jax = resnet50(key=jax.random.PRNGKey(33), make_with_state=False)
    resnet_torch = t_resnet50(weights=ResNet50_Weights.DEFAULT)
    state_dict = resnet_torch.state_dict()

    s2p.start_conversion(resnet_jax, state_dict)


if __name__ == "__main__":
    convert_resnet()
