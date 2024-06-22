import equinox as eqx
import jax
from statedict2pytree.utils.utils_pytree import get_node, replace_node


def test_replace_node():
    class MyModel(eqx.Module):
        layer1: eqx.nn.Linear
        layer2: eqx.nn.Linear

    model = MyModel(
        layer1=eqx.nn.Linear(10, 20, key=jax.random.key(0)),
        layer2=eqx.nn.Linear(20, 5, key=jax.random.key(1)),
    )
    new_weight = jax.numpy.ones((20, 10))
    updated_model = replace_node(model, ["layer1", "weight"], new_weight)
    assert (updated_model.layer1.weight == new_weight).all()


def test_get_node():
    class MyModel(eqx.Module):
        layer1: eqx.nn.Linear
        layer2: eqx.nn.Linear

    model = MyModel(
        layer1=eqx.nn.Linear(10, 20, key=jax.random.key(0)),
        layer2=eqx.nn.Linear(20, 5, key=jax.random.key(1)),
    )
    layer1_weight = get_node(model, ["layer1", "weight"])
    assert layer1_weight is not None
    assert layer1_weight.shape == (20, 10)
    nonexistent_node = get_node(model, ["layer3"])
    assert nonexistent_node is None
