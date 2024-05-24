import json

import jax
import torch
from jaxonmodels.transformers.llama.llama3 import LLaMA
from jaxonmodels.transformers.llama.model_args import LLaMAModelArgs
from memory_profiler import profile


@profile
def main():
    state_dict = torch.load("Meta-Llama-3-8B/consolidated.00.pth")

    key_len = len(state_dict)
    print("key_len", key_len)
    keys_to_delete = [
        key for i, key in enumerate(state_dict.keys()) if i < int(key_len * 0.9999)
    ]

    for key in keys_to_delete:
        del state_dict[key]

    for key in state_dict.keys():
        if torch.is_tensor(state_dict[key]):
            state_dict[key] = state_dict[key].to(torch.float16)

    print("key_len after", len(state_dict))
    with open("Meta-Llama-3-8B/params.json", "r") as f:
        params = json.load(f)

    model_args = LLaMAModelArgs(**params)
    key = jax.random.PRNGKey(21)
    model = LLaMA(model_args, key=key)


if __name__ == "__main__":
    main()
