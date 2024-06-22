import equinox as eqx
import jax
import numpy as np
import statedict2pytree as s2p
import torch


def test_linear():
    in_features = 10
    out_features = 10

    class J(eqx.Module):
        linear: eqx.nn.Linear
        norm: eqx.nn.BatchNorm

        def __init__(self):
            self.linear = eqx.nn.Linear(
                in_features, out_features, key=jax.random.PRNGKey(30)
            )
            self.norm = eqx.nn.BatchNorm(input_size=out_features, axis_name="batch")

    class T(torch.nn.Module):
        def __init__(self) -> None:
            super(T, self).__init__()
            self.linear = torch.nn.Linear(in_features, out_features)
            self.norm = torch.nn.BatchNorm1d(out_features)

    jax_model = J()
    torch_model = T()
    state_dict = torch_model.state_dict()

    model, state = s2p.autoconvert_state_dict_to_pytree(
        pytree=jax_model, state_dict=state_dict
    )

    assert np.allclose(
        np.array(model.linear.weight), torch_model.linear.weight.detach().numpy()
    )
    assert np.allclose(
        np.array(model.linear.bias), torch_model.linear.bias.detach().numpy()
    )

    assert np.allclose(
        np.array(model.norm.weight), torch_model.norm.weight.detach().numpy()
    )
    assert np.allclose(
        np.array(model.norm.bias), torch_model.norm.bias.detach().numpy()
    )
