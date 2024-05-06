import equinox as eqx
import jax
import torch
import torch2jax as t2j


def test_mlp():
    in_size = 784
    out_size = 10
    width_size = 64
    depth = 2
    key = jax.random.PRNGKey(22)

    class EqxMLP(eqx.Module):
        mlp: eqx.nn.MLP
        batch_norm: eqx.nn.BatchNorm

        def __init__(self, in_size, out_size, width_size, depth, key):
            self.mlp = eqx.nn.MLP(in_size, out_size, width_size, depth, key=key)
            self.batch_norm = eqx.nn.BatchNorm(out_size, axis_name="batch")

        def __call__(self, x, state):
            return self.batch_norm(self.mlp(x), state)

    jax_model = EqxMLP(in_size, out_size, width_size, depth, key)

    class TorchMLP(torch.nn.Module):
        def __init__(self, in_size, out_size, width_size, depth):
            super(TorchMLP, self).__init__()
            self.layers = torch.nn.ModuleList()
            self.layers.append(torch.nn.Linear(in_size, width_size))
            for _ in range(depth - 1):
                self.layers.append(torch.nn.Linear(width_size, width_size))
            self.layers.append(torch.nn.Linear(width_size, out_size))
            self.batch_norm = torch.nn.BatchNorm1d(out_size)

        def forward(self, x):
            for layer in self.layers[:-1]:
                x = torch.relu(layer(x))
            x = self.batch_norm(self.layers[-1](x))
            return x

    torch_model = TorchMLP(in_size, out_size, width_size, depth)
    state_dict = torch_model.state_dict()
    t2j.convert(jax_model, state_dict)


if __name__ == "__main__":
    test_mlp()
