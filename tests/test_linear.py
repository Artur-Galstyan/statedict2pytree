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

        def __init__(self):
            self.linear = eqx.nn.Linear(
                in_features, out_features, key=jax.random.PRNGKey(30)
            )

    class T(torch.nn.Module):
        def __init__(self) -> None:
            super(T, self).__init__()
            self.linear = torch.nn.Linear(in_features, out_features)

    jax_model = J()
    torch_model = T()
    state_dict = torch_model.state_dict()

    jax_fields = s2p.pytree_to_fields(jax_model)
    torch_fields = s2p.state_dict_to_fields(state_dict)

    model, state = s2p.convert(
        jax_fields, torch_fields, pytree=jax_model, state_dict=state_dict
    )

    assert np.allclose(
        np.array(model.linear.weight), torch_model.linear.weight.detach().numpy()
    )
    assert np.allclose(
        np.array(model.linear.bias), torch_model.linear.bias.detach().numpy()
    )


if __name__ == "__main__":
    test_linear()
