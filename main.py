import equinox as eqx
import jax
import jax.numpy as jnp


mlp = eqx.nn.MLP(
    in_size=128,
    out_size=128,
    depth=3,
    width_size=128,
    key=jax.random.key(2),
    dtype=jnp.float32,
)
print(mlp.layers[-1].weight.dtype)
new_final_layer = eqx.nn.Linear(
    in_features=128, out_features=128, key=jax.random.key(4), dtype=jnp.float16
)
where = lambda m: m.layers[-1]
mlp = eqx.tree_at(where, mlp, new_final_layer)

print(mlp.layers[-1].weight.dtype)
