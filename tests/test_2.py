import equinox as eqx
import jax


def test():
    model, state = eqx.nn.make_with_state(eqx.nn.MLP)(
        3, 3, 3, 3, key=jax.random.PRNGKey(33)
    )
    print(model)
    print(state)
    breakpoint()


if __name__ == "__main__":
    test()
