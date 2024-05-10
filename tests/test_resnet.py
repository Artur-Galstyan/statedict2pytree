import jax
import torch2jax as t2j
from tests.resnet import resnet50
from torchvision.models import resnet50 as t_resnet50, ResNet50_Weights


def test_resnet():
    resnet_jax = resnet50(key=jax.random.PRNGKey(33), make_with_state=False)
    resnet_torch = t_resnet50(weights=ResNet50_Weights.DEFAULT)
    state_dict = resnet_torch.state_dict()

    t2j.convert(resnet_jax, state_dict)


if __name__ == "__main__":
    test_resnet()
