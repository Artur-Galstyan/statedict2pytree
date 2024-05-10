import equinox as eqx
import jax


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

    model, state = eqx.nn.make_with_state(EqxMLP)(
        in_size, out_size, width_size, depth, key
    )

    eqx.tree_serialise_leaves("test1.eqx", (model, state))

    new_model, new_state = eqx.tree_deserialise_leaves("test1.eqx", (model, state))


if __name__ == "__main__":
    test_mlp()
