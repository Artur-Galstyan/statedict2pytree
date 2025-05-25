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

    model, state = eqx.nn.make_with_state(J)()
    print(model, state)
    torch_model = T()
    state_dict = torch_model.state_dict()

    torchfields = s2p.state_dict_to_fields(state_dict)
    torchfields = s2p.move_running_fields_to_the_end(torchfields)

    jaxfields, state_indices = s2p.pytree_to_fields(
        (model, state), filter=s2p.is_numerical
    )

    model, state = s2p.convert(
        state_dict, (model, state), jaxfields, state_indices, torchfields
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
