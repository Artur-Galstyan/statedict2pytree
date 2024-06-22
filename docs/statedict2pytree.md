# StateDict to PyTree Converter

This module provides the core functionality for converting PyTorch state dictionaries to JAX pytrees.

## Main Conversion Functions

::: statedict2pytree.statedict2pytree
    handler: python
    selection:
      members:
        - autoconvert_state_dict_to_pytree
        - autoconvert_from_paths
        - convert_from_path
        - convert_from_pytree_and_state_dict
    rendering:
      show_root_heading: false
      show_source: true

## Server Initialization Functions

::: statedict2pytree.statedict2pytree
    handler: python
    selection:
      members:
        - start_conversion_from_paths
        - start_conversion_from_pytree_and_state_dict
    rendering:
      show_root_heading: false
      show_source: true

## Flask Server Routes

::: statedict2pytree.statedict2pytree
    handler: python
    selection:
      members:
        - _get_torch_fields
        - _get_jax_fields
        - _convert_torch_to_jax
        - make_anthropic_request
    rendering:
      show_root_heading: false
      show_source: true
