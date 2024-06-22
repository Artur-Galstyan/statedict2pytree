import equinox as eqx
import jax
import torch


def test_long_models():
    n_layers = 20

    class Model(eqx.Module):
        layers: list

        def __init__(self):
            key, *subkeys = jax.random.split(jax.random.key(42), n_layers + 1)
            self.layers = []
            for l in range(n_layers):
                self.layers.append(
                    eqx.nn.Linear(
                        in_features=l % 5 + 1,
                        out_features=l % 5 + 1,
                        key=subkeys[l],
                        use_bias=False,
                    )
                )

    class TorchModel(torch.nn.Module):
        def __init__(self):
            super(TorchModel, self).__init__()
            self.layers = torch.nn.Sequential(
                *[
                    torch.nn.Linear(
                        in_features=5 - (l % 5 + 1),
                        out_features=5 - (l % 5 + 1),
                        bias=False,
                    )
                    for l in range(int(n_layers * 0.75))
                ]
            )

        def forward(self, x):
            return x

    pytree = Model()
    state_dict = TorchModel().state_dict()
    # start_conversion_from_pytree_and_state_dict(pytree, state_dict)


if __name__ == "__main__":
    test_long_models()
