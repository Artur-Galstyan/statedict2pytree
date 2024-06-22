import json

import equinox as eqx
import jax
import jax.numpy as jnp
import tokenizer as tk
from jaxonmodels.transformers.llama.llama3 import LLaMA
from jaxonmodels.transformers.llama.model_args import LLaMAModelArgs
from jaxtyping import PyTree
from loguru import logger
from tqdm import tqdm


with open("Meta-Llama-3-8B/params.json", "r") as f:
    params = json.load(f)

model_args = LLaMAModelArgs(**params)
model_args.precision = "quarter"
key = jax.random.PRNGKey(21)
logger.info("Creating JAX model...")
model, state = eqx.nn.make_with_state(LLaMA)(model_args, key=key)
logger.info("Done.")
tokenizer_model_path = "Meta-Llama-3-8B/tokenizer.model"
tokenizer = tk.Tokenizer(tokenizer_model_path)

prompt = "What is the meaning of life?"
tokens = tokenizer.encode(prompt, bos=True, eos=False)
model_args.vocab_size = tokenizer.n_words

tokens = tokenizer.encode(prompt, bos=True, eos=False)
# tokens += [0 for _ in range(model_args.max_seq_len - len(tokens))]
tokens = [int(t) for t in tokens]


@eqx.filter_jit
def generate_text(
    model: PyTree,
    tokens,
    max_new_tokens: int,
    vocab_size: int,
    state: eqx.nn.State,
    random_key_seed: int = 0,
):
    key = jax.random.PRNGKey(random_key_seed)

    def _body(carry, token):
        key, state = carry
        key, subkey = jax.random.split(key)
        x = jnp.array([token], dtype=jnp.int32)
        logits, state = model(x, state=state, key=key, mask="causal")
        logits = logits[-1, :]
        probs = jax.nn.softmax(logits, axis=-1)

        next_token = jax.random.choice(
            subkey,
            jnp.arange(len(probs)),
            p=probs,
        )
        next_token = jnp.array(next_token, dtype=jnp.int32).reshape((1,))
        return (key, state), next_token

    init = (jax.random.key(random_key_seed), state)
    carry, ys = jax.lax.scan(_body, init, jnp.array(tokens))
    _, state = carry
    return ys, state


# new_tokens, state = generate_text(
#     model=model,
#     tokens=tokens,
#     max_new_tokens=1,
#     vocab_size=model_args.vocab_size,
#     state=state,
# )

# breakpoint()
# new_tokens = [int(t) for t in tqdm(new_tokens[0])]  # pyright: ignore
# logger.info("Decoding tokens...")
# print(tokenizer.decode(new_tokens))

model, state = eqx.nn.make_with_state(LLaMA)(model_args, key=jax.random.key(2))
model, state = eqx.tree_deserialise_leaves("llama3-8b.eqx", (model, state))

tokens, state = generate_text(
    model=model,
    tokens=tokens,
    max_new_tokens=4,
    vocab_size=model_args.vocab_size,
    state=state,
)

new_tokens = [int(t) for t in tqdm(new_tokens[0])]  # pyright: ignore
logger.info("Decoding tokens...")
print(tokenizer.decode(new_tokens))
