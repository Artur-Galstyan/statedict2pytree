# Quickstart Guide

## Installation

To install StateDict2PyTree, run:

```bash
pip install statedict2pytree
```

## Basic Usage
StateDict2PyTree provides two main ways to convert PyTorch state dicts to JAX pytrees:

1. Using paths to chunked files
2. Using in-memory structures

### Using Paths to Chunked Files
This method is useful for large models that may not fit in memory.

```python
import statedict2pytree as s2p
from statedict2pytree.utils.utils_state_dict import chunkify_state_dict
from statedict2pytree.utils.utils_pytree import chunkify_pytree

# Chunkify your PyTorch state dict
chunkify_state_dict(torch_state_dict, "path/to/tmp/dir")

# Chunkify your JAX model
jax_model = create_your_jax_model()
chunkify_pytree(jax_model, "path/to/tmp/dir")

# Start the conversion server
s2p.start_conversion_from_paths("path/to/tmp/dir", "path/to/tmp/dir")

# Cancel the server using CTRL+C. This will continue the code execution

# Load model
```

### Using In-Memory Structures
This method is suitable for smaller models that can fit in memory.

```python
import statedict2pytree as s2p

jax_model = create_your_jax_model()
torch_state_dict = load_your_torch_state_dict()

s2p.start_conversion_from_pytree_and_state_dict(jax_model, torch_state_dict)
```

After running either of these methods, you can access the conversion UI by opening a web browser and navigating to http://localhost:5500

### Advanced Usage
For more advanced usage and API details, please refer to the API Reference.
