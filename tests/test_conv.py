import equinox as eqx
import jax
import numpy as np
import statedict2pytree as s2p
import torch


def test_conv():
    in_channels = 8
    out_channels = 8
    kernel_size = 4
    stride = 2
    padding = 1

    class J(eqx.Module):
        conv: eqx.nn.Conv2d

        def __init__(self):
            self.conv = eqx.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                key=jax.random.PRNGKey(22),
            )

    class T(torch.nn.Module):
        def __init__(self) -> None:
            super(T, self).__init__()
            self.conv = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )

    jax_model = J()
    torch_model = T()
    state_dict = torch_model.state_dict()

    model = s2p.autoconvert(jax_model, state_dict)

    assert np.allclose(
        np.array(model.conv.weight), torch_model.conv.weight.detach().numpy()
    )
    if torch_model.conv.bias is not None:
        assert np.allclose(
            np.array(model.conv.bias),
            torch_model.conv.bias.detach().numpy().reshape(model.conv.bias.shape),
        )
