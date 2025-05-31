# statedict2pytree


## Update:

For examples for `statedict2pytree`, check out my other repository [jaxonmodels](https://github.com/Artur-Galstyan/jaxonmodels).

## Docs

Docs can be found [here](https://artur-galstyan.github.io/statedict2pytree/).


## Info

`statedict2pytree` is a powerful tool for converting PyTorch state dictionaries to JAX pytrees, specifically for Equinox


## Installation

```bash
pip install statedict2pytree
```

The goal of this package is to simplify the conversion from PyTorch models into JAX PyTrees (which can be used e.g. in Equinox). The way this works is by putting both models side my side and aligning the weights in the right order. Then, all statedict2pytree is doing, is iterating over both lists and matching the weight matrices.

Usually, if you _declared the fields in the same order as in the PyTorch model_, you don't have to rearrange anything -- but the option is there if you need it.


## Shape Matching? What's that?

Currently, there is no sophisticated shape matching in place. Two matrices are considered "matching" if the product of their shape match. For example:

(8, 1, 1) and (8, ) match, because (8 _ 1 _ 1 = 8)


### Disclaimer

Some of the docstrings and the docs have been written with the help of
Claude.

