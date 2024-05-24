# statedict2pytree

![statedict2pytree](torch2jax.png "A ResNet demo")

The goal of this package is to simplify the conversion from PyTorch models into JAX PyTrees (which can be used e.g. in Equinox). The way this works is by putting both models side my side and aligning the weights in the right order. Then, all statedict2pytree is doing, is iterating over both lists and matching the weight matrices.

Usually, if you _declared the fields in the same order as in the PyTorch model_, you don't have to rearrange anything -- but the option is there if you need it.

(Theoretically, you can rearrange the model in any way you like - e.g. last layer as the first layer - as long as the shapes match!)

## Shape Matching? What's that?

Currently, there is no sophisticated shape matching in place. Two matrices are considered "matching" if the product of their shape match. For example:

1. (8, 1, 1) and (8, ) match, because (8 _ 1 _ 1 = 8)

## Get Started

### Installation

Run

```bash
pip install statedict2pytree

```

### Docs

Documentation will appear as soon as I have all the necessary features implemented. Until then, check out the "main.py" file for a better example.
