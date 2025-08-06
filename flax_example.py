import torch
from flax import nnx
from statedict2pytree import autoconvert, pytree_to_fields


class TorchModel(torch.nn.Module):
    def __init__(self, din, dout):
        super(TorchModel, self).__init__()
        self.linear = torch.nn.Linear(din, dout, bias=False)
        self.linear.weight.data = torch.ones_like(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


class Model(nnx.Module):
    def __init__(self, din, dout, rngs: nnx.Rngs):
        self.linear = nnx.Linear(din, dout, rngs=rngs, use_bias=False)

    def __call__(self, x):
        return self.linear(x)


flax_model = Model(2, 64, rngs=nnx.Rngs(0))
torch_model = TorchModel(2, 64)

pt_fields = pytree_to_fields(flax_model)
print(pt_fields)

flax_model = autoconvert(flax_model, torch_model.state_dict())

print(flax_model)
print(flax_model.linear.kernel.value)
