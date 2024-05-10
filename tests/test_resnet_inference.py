import functools as ft
import json
import urllib

import equinox as eqx
import jax
import jax.numpy as jnp
import torch
from PIL import Image
from tests.resnet import resnet50
from torchvision import transforms
from torchvision.models import resnet50 as t_resnet50, ResNet50_Weights


def test_resnet():
    resnet_jax = resnet50(key=jax.random.PRNGKey(33), make_with_state=False)
    resnet_torch = t_resnet50(weights=ResNet50_Weights.DEFAULT)

    img_name = "doggo.jpeg"

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = Image.open(img_name)
    img_t = transform(img)
    print(img_t.shape)  # pyright: ignore
    batch_t = torch.unsqueeze(img_t, 0)  # pyright:ignore

    # Predict
    with torch.no_grad():
        output = resnet_torch(batch_t)
        print(output.shape)  # pyright: ignore
        _, predicted = torch.max(output, 1)

    print(
        f"Predicted: {predicted.item()}"
    )  # Outputs the ImageNet class index of the prediction

    url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    with urllib.request.urlopen(url) as url:
        imagenet_labels = json.loads(url.read().decode())

    label = imagenet_labels[str(predicted.item())][1]
    print(f"Label for index {predicted.item()}: {label}")

    identity = lambda x: x
    model_callable = ft.partial(identity, resnet_jax)
    model, state = eqx.nn.make_with_state(model_callable)()

    model, state = eqx.tree_deserialise_leaves("model.eqx", (model, state))

    jax_batch = jnp.array(batch_t.numpy())
    out, state = eqx.filter_vmap(
        model, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
    )(jax_batch, state)
    print(f"{out.shape}")

    label = imagenet_labels[str(jnp.argmax(out))][1]
    print(f"Label for index {jnp.argmax(out)}: {label}")


if __name__ == "__main__":
    test_resnet()
